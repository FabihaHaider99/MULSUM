"""End-to-end smoke test for MULSUM: tiny train -> merge LoRA in memory -> generate one summary.

Run from the MULSUM repo root:

    python smoke_test.py
    python smoke_test.py --max-input-len 1024            # smaller GPU
    python smoke_test.py --skip-train --adapter checkpoints/smoke_<ts>   # reuse a finished run

Stages (each prints PASS/FAIL and stops at the first failure):
  1. train   24 train / 8 val samples, 12 steps -> checkpoints/smoke_<ts>/
  2. merge   load_pretrained_model(): Vicuna + stage-1 projector + non_lora_trainables + LoRA, merged
  3. infer   first --n-test test rows (default 5), features from the test LMDB, 64 new tokens, greedy

Self-contained: only imports from the repo's src/. Saving to HF is NOT tested here
(that is merge_and_push.py); this checks that what training produced loads and generates.
"""
import argparse
import datetime
import gc
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}".rstrip())
    if not ok:
        sys.exit(1)


def stage_train(cfg, args):
    steps = max(1, args.n_train // args.batch_size)
    out_dir = os.path.join("checkpoints", "smoke_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    cfg["training"].update(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=max(1, steps // 2),
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=1,
        load_best_model_at_end=False,
        early_stopping_patience=None,
        dataloader_num_workers=2,       # prefetch_factor must be None if workers == 0
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=False,
    )
    cfg["data"]["train_subset_size"] = args.n_train
    cfg["data"]["validation_subset_size"] = args.n_val

    # Small copies of the CSVs with the column names dataset.py expects
    d = cfg["data"]
    tmp_files = [os.path.join("data", "_smoke_train.csv")]
    prep_csv(d["data_path"], tmp_files[0], args.n_train)
    d["data_path"] = tmp_files[0]
    if d.get("validation_path") and os.path.exists(d["validation_path"]):
        tmp_files.append(os.path.join("data", "_smoke_val.csv"))
        prep_csv(d["validation_path"], tmp_files[1], args.n_val)
        d["validation_path"] = tmp_files[1]
    if args.max_input_len:
        cfg["data"]["max_input_len"] = args.max_input_len
    cfg["wandb"]["enabled"] = False

    tmp_cfg = os.path.join("configs", "_smoke_config.json")
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    from src.train.train_image_text import train_image_text_summarization

    try:
        train_image_text_summarization(tmp_cfg)
    finally:
        for p in [tmp_cfg] + tmp_files:
            if os.path.exists(p):
                os.remove(p)
    return out_dir


def check_train_outputs(out_dir):
    import pandas as pd

    for name in ("adapter_config.json", "non_lora_trainables.bin", "loss_history.csv"):
        check(f"train output: {name}", os.path.exists(os.path.join(out_dir, name)))
    df = pd.read_csv(os.path.join(out_dir, "loss_history.csv"))
    print(df.to_string(index=False))
    ev = df["eval_loss"].dropna()
    check("eval_loss recorded and finite", len(ev) > 0 and all(math.isfinite(x) for x in ev),
          f"(last={ev.iloc[-1]:.4f})" if len(ev) else "")


def load_merged_on_gpu(model_base, stage1, adapter):
    """Same steps as builder.load_pretrained_model, but the 7B weights go straight to the GPU.

    load_pretrained_model() builds the model in CPU RAM first, which gets the process
    OOM-killed ("Killed") on a 16 GiB machine such as g6.xlarge.
    """
    import torch
    from transformers import AutoTokenizer

    from src.model import MulsumLlamaForCausalLM
    from src.model.builder import get_local_model_path, load_lora

    base = get_local_model_path(model_base)
    local = {"local_files_only": True} if os.path.exists(base) else {}
    tok = AutoTokenizer.from_pretrained(base, **local)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = MulsumLlamaForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0}, **local)
    model.get_model().initialize_vision_modules(argparse.Namespace(pretrain_mm_mlp_adapter=stage1))
    model = load_lora(model, adapter).merge_and_unload()
    return tok, model.to("cuda", torch.float16)


def stage_merge(cfg, adapter):
    import torch
    m = cfg["model"]
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory in use before merge: {torch.cuda.memory_allocated() / 2**30:.1f} GiB "
          "(should be ~0; if it is >10, rerun with --skip-train in a fresh process)")
    tok, model = load_merged_on_gpu(m["model_name_or_path"], m["pretrain_mm_mlp_adapter"], adapter)
    model.eval()
    proj = model.get_model().mm_projector
    check("merged model has mm_projector",
          tuple(proj.weight.shape) == (model.config.hidden_size, cfg["data"]["image_feature_dim"]),
          f"(weight {tuple(proj.weight.shape)})")
    return tok, model


def stage_infer(cfg, tok, model, n_test=5):
    import pandas as pd
    import torch
    from src.constants import IMAGE_TOKEN_INDEX
    from src.conversation import SeparatorStyle, conv_templates
    from src.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
    from src.prompt_manager import format_prompt_with_images
    from src.train.dataset import load_image_features, open_lmdb

    d = cfg["data"]
    df = pd.read_csv(d["test_path"]).head(n_test)
    env = open_lmdb(d["test_lmdb_path"])
    version = cfg["model"].get("version", "v1")
    results = []

    for i, (_, row) in enumerate(df.iterrows()):
        paths = row.get("Image_Paths", [])
        if isinstance(paths, str):
            paths = json.loads(paths)
        if not isinstance(paths, list):
            paths = []

        feats, mask, used = load_image_features(env, paths, d["max_images"], d["image_feature_dim"])
        n = int(mask.sum().item())
        check(f"[{i}] image features loaded", feats.shape == (d["max_images"], d["image_feature_dim"]),
              f"(shape {tuple(feats.shape)}, {n} real images)")

        body = pick(row, ALIASES["Body"])
        check(f"[{i}] test row has article text", bool(body), f"(columns: {list(row.index)})")
        ids = tok.encode(body, add_special_tokens=False)[:1500]
        text = tok.decode(ids, skip_special_tokens=True)
        conv = conv_templates.get(version, conv_templates["v1"]).copy()
        conv.append_message(conv.roles[0], format_prompt_with_images(text, n, mode="inference"))
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                images=feats.unsqueeze(0).to("cuda", torch.float16),
                do_sample=False, num_beams=1, max_new_tokens=64, use_cache=True,
                stopping_criteria=[KeywordsStoppingCriteria([stop], tok, input_ids)],
            )
        summary = tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        check(f"[{i}] generated non-empty text", len(summary) > 0)
        reference = pick(row, ALIASES["Summary"])
        print(f"\n[{i}] summary: {summary}")
        if reference:
            print(f"[{i}] reference: {reference[:300]}")

        # Same columns as results/inference_results.csv from image_text_inference.py
        results.append({
            "id": row.get("id", row.get("Article_ID", i)),
            "text": text[:500] + ("..." if len(text) > 500 else ""),
            "image_paths": used,
            "generated_summary": summary,
            "reference_summary": reference,
        })

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "smoke_test.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    check("saved results", os.path.exists(out_csv), f"({out_csv}, {len(results)} rows)")


# dataset.py hard-requires columns named exactly "Body" and "Summary"
ALIASES = {
    "Body": ["Body", "body", "Text", "text", "Article_Text", "Content", "content"],
    "Summary": ["Summary", "summary", "Reference_Summary", "reference_summary", "Reference", "reference"],
}


def prep_csv(src, dst, n):
    """Write the first n rows of src to dst, renaming article/summary columns to Body/Summary."""
    import pandas as pd

    df = pd.read_csv(src).head(n)
    for target, names in ALIASES.items():
        if target not in df.columns:
            found = next((c for c in names if c in df.columns), None)
            if found is None:
                sys.exit(f"{src}: no column for '{target}'. Columns are {list(df.columns)}")
            print(f"{src}: using column '{found}' as '{target}'")
            df = df.rename(columns={found: target})
    df.to_csv(dst, index=False)


def pick(row, names, default=""):
    """First non-empty value among the candidate column names."""
    for n in names:
        if n in row and str(row[n]) not in ("nan", ""):
            return str(row[n])
    return default


def fix_stage1(cfg):
    """Fall back to a random-init projector if the stage-1 file doesn't fit this repo.

    The repo builds mm_projector = nn.Linear(image_feature_dim, hidden) and loads keys
    '...mm_projector.weight/bias'. A LLaVA-style MLP projector ('0.weight', '2.weight')
    fails with "Missing key(s) in state_dict: weight, bias".
    """
    import torch

    path = cfg["model"].get("pretrain_mm_mlp_adapter")
    if not path:
        return
    dim = cfg["data"]["image_feature_dim"]
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        w = {k.split("mm_projector.")[1]: v for k, v in sd.items() if "mm_projector" in k}
        ok = sorted(w) == ["bias", "weight"] and w["weight"].shape[1] == dim
        found = {k: tuple(v.shape) for k, v in w.items()} or sorted(sd)
    except Exception as e:
        ok, found = False, repr(e)
    if not ok:
        print(f"WARNING: stage-1 projector {path} does not fit this repo (found {found}; "
              f"need weight/bias with in_features={dim}). "
              "Using a randomly initialised projector for this smoke test.")
        cfg["model"]["pretrain_mm_mlp_adapter"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("configs", "image_text_config.json"))
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-input-len", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=5, help="test samples to summarize (saved to results/smoke_test.csv)")
    ap.add_argument("--skip-train", action="store_true", help="reuse --adapter instead of training")
    ap.add_argument("--adapter", default=None, help="finished run dir (with --skip-train)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    fix_stage1(cfg)

    if args.skip_train:
        if not args.adapter:
            ap.error("--skip-train needs --adapter")
        out_dir = args.adapter
    else:
        out_dir = stage_train(cfg, args)
    check_train_outputs(out_dir)

    tok, model = stage_merge(cfg, out_dir)
    stage_infer(cfg, tok, model, args.n_test)
    print("\nAll smoke checks passed. Output dir:", out_dir)


if __name__ == "__main__":
    main()