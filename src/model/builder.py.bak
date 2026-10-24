import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from src.model import *
from peft import PeftModel

def load_lora(model, lora_path):
    # Check if this is a local checkpoint directory that has numbered checkpoints
    if os.path.isdir(lora_path) and not os.path.exists(os.path.join(lora_path, 'adapter_config.json')):
        # Look for the latest checkpoint or checkpoint-2000
        checkpoints = [d for d in os.listdir(lora_path) if d.startswith('checkpoint-')]
        if not checkpoints:
            raise ValueError(f"No checkpoint directories found in {lora_path}")
        
        # Sort checkpoints by number to get the latest one
        checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
        lora_path = os.path.join(lora_path, checkpoints[0])
        print(f"Using latest checkpoint: {lora_path}")
    
    # Load non-LoRA trainable weights
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
    print('Loading LoRA weights...')
    # Explicitly set local_files_only=True to prevent trying to download from Hub
    model = PeftModel.from_pretrained(model, lora_path, local_files_only=True)
    return model

def get_local_model_path(model_base):
    """Get local model path if it exists, otherwise return the original path"""
    # Check if it's already a local path
    if os.path.exists(model_base):
        return model_base
    
    # Check for local models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    local_model_path = os.path.join(models_dir, model_base.replace("/", "_"))
    
    if os.path.exists(local_model_path):
        print(f"Using local model from: {local_model_path}")
        return local_model_path
    else:
        print(f"Local model not found at {local_model_path}, using online: {model_base}")
        return model_base

def load_pretrained_model(args, stage2=None, stage3=None):
    kwargs = {'torch_dtype': torch.float16}

    model_base = get_local_model_path(args.model_base)
    
    # Configure tokenizer loading
    load_kwargs = {}
    if os.path.exists(model_base):
        load_kwargs['local_files_only'] = True
    
    tokenizer = AutoTokenizer.from_pretrained(model_base, **load_kwargs)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add local_files_only for model loading too
    model_kwargs = {'low_cpu_mem_usage': True, **kwargs}
    if os.path.exists(model_base):
        model_kwargs['local_files_only'] = True
    
    model = MulsumLlamaForCausalLM.from_pretrained(model_base, **model_kwargs)
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))


    # load stage1:
    model.get_model().initialize_vision_modules(args)

    if stage2 is not None:
        print('Loading stage2 weights...')
        model = load_lora(model, stage2)
        print('Merging stage2 weights...')
        model = model.merge_and_unload()
        if stage3 is not None:
            print('Loading stage3 weights...')
            model = load_lora(model, stage3)
            print('Merging stage3 weights...')
            model = model.merge_and_unload()


    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
