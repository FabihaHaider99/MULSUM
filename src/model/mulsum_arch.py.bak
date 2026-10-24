import os
import torch
import torch.nn as nn
from src.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig


class MulsumMetaModel:

    def get_local_model_path(self, model_name):
        """Get local model path if it exists, otherwise return the original path"""
        # Check if it's already a local path
        if os.path.exists(model_name):
            return model_name
        
        # Check for local models directory relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "..", "..", "models")
        local_model_path = os.path.join(models_dir, model_name.replace("/", "_"))
        
        if os.path.exists(local_model_path):
            print(f"Using local tokenizer from: {local_model_path}")
            return local_model_path
        else:
            print(f"Local tokenizer not found, using online: {model_name}")
            return model_name

    def initialize_vision_modules(self, model_args):
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        trainable_mm_projector = getattr(model_args, "trainable_mm_projector", False)
        
        # Use local model if available
        tokenizer_path = self.get_local_model_path("lmsys/vicuna-7b-v1.5")
        tokenizer_kwargs = {}
        
        if os.path.exists(tokenizer_path) and tokenizer_path != "lmsys/vicuna-7b-v1.5":
            tokenizer_kwargs['local_files_only'] = True
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        
        # Set pad token if not already set (newer transformers handle this better)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(768, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu', weights_only=True)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("load mlp:", pretrain_mm_mlp_adapter)
        
        # Set requires_grad based on trainable_mm_projector flag
        for param in self.mm_projector.parameters():
            param.requires_grad = trainable_mm_projector
            
        # Store the flag in the config for reference during checkpoint saving
        if not hasattr(self.config, 'trainable_mm_projector'):
            self.config.trainable_mm_projector = trainable_mm_projector
        print(f"MM Projector is {'trainable' if trainable_mm_projector else 'frozen'}")


class MulsumMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        # Return early if no images or only a single token in input_ids (for generation)
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Process image features - shape is (batch_size, num_images, feature_dim)
        batch_size, num_images, feature_dim = images.shape
        # Reshape to (batch_size * num_images, feature_dim) for processing
        flat_images = images.reshape(-1, feature_dim)
        # Project the features using mm_projector
        flat_image_features = self.get_model().mm_projector(flat_images)
        # Reshape back to (batch_size, num_images, hidden_dim)
        image_features = flat_image_features.reshape(batch_size, num_images, -1)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Remove padding tokens using attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # No image tokens found, just use text embeddings
                cur_input_embeds = self.get_model().get_input_embeddings()(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            # Find positions of image tokens
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # Get available images for this batch item
            batch_images = image_features[batch_idx]  # shape: (num_images, hidden_dim)
            available_images = min(batch_images.shape[0], num_images)
            
            # Interleave text segments and images properly
            for i in range(len(cur_input_embeds_no_im)):
                # Add text segment
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                
                # Add corresponding image if available (except for the last text segment)
                if i < available_images:
                    # Add single image embedding at this position
                    single_image = batch_images[i:i+1]  # Keep batch dimension: (1, hidden_dim)
                    cur_new_input_embeds.append(single_image)
                    cur_new_labels.append(torch.full((1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine and pad sequences
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        fake_input_ids = None
        return fake_input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
