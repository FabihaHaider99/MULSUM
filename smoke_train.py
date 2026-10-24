"""Smoke test: run a handful of MULSUM training steps on a tiny subset.

Run from the MULSUM repo root (config paths are relative to it):

    python smoke_train.py
    python smoke_train.py --n-train 24 --batch-size 2 --max-input-len 1024

Output goes to checkpoints/smoke_<timestamp>/ (a fresh dir, so the training
script's auto-resume from checkpoint-* never kicks in).
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


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
    ap.add_argument("--n-train", type=int, default=24, help="training samples to use")
    ap.add_argument("--n-val", type=int, default=8, help="validation samples to use")
    ap.add_argument("--batch-size", type=int, default=2, help="per-device batch size")
    ap.add_argument("--max-input-len", type=int, default=None,
                    help="override data.max_input_len (e.g. 1024 for a smaller GPU)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    fix_stage1(cfg)

    steps = max(1, args.n_train // args.batch_size)
    if steps < 10:
        print(f"NOTE: {steps} steps planned. train_image_text.py hardcodes logging_steps=10, "
              "so no train loss will be logged below 10 steps (eval loss still is).")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("checkpoints", f"smoke_{stamp}")

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
        # prefetch_factor must be None when num_workers == 0, so keep workers > 0
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=False,
    )
    cfg["data"]["train_subset_size"] = args.n_train
    cfg["data"]["validation_subset_size"] = args.n_val
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
        os.remove(tmp_cfg)

    print("\n=== smoke run finished ===")
    print("output dir:", out_dir)
    for name in sorted(os.listdir(out_dir)):
        print("  ", name)

    loss_csv = os.path.join(out_dir, "loss_history.csv")
    if os.path.exists(loss_csv):
        import pandas as pd
        print("\nloss_history.csv:")
        print(pd.read_csv(loss_csv).to_string(index=False))

    print("\nNext: python merge_and_push.py --adapter", out_dir, "--out checkpoints/merged_smoke")


if __name__ == "__main__":
    main()