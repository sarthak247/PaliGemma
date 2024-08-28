import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SigLipVisionConfig, SiglipVisionModel

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # Since shape of key_cache is [batch, num_heads_kv, seq_len, head_dim]
            return self.key_cache[0].shape[-2] # returns seq_len
    
    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise, we concatenate the new keys with the existing ones.
            # each tensor has shape: [batch, num_heads_kv, seq_lenm, head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)

        # and then we return all the existing keys + the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
            

class GemmaConfig():
    def __init__(
            self,
            vocab_size, # size of vocabulary
            hidden_size, # size of embedding vector of each token
            intermediate_size, # size of FF layer
            num_hidden_layers, # num of layers
            num_attention_heads, # attention_heads for Queries
            num_key_value_heads, # attention_heads for KV (they are different due to grouped query attention)
            head_dim = 256, # dimenions of each head
            max_position_embeddings = 8192, # max number of positions (needed for ROPE)
            rms_norm_eps = 1e-6, # RMSNorm
            rope_theta = 10000.0, # ROPE parameter
            attention_bias = False, # if we need bias in the attention matrices
            attention_dropout = 0.0, # Dropout
            pad_token_id = None,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size =  hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings = 2048, base = 10000, device = None):
        super().__init__()

        self.dim = dim #set to the head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, 3, ...dim //2
        inv_freq = 1.0/ (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len = None):
        # x: [batch, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        # Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [batch, head_dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [batch, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type 
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [batch, head_dim // 2, 1] @ [batch, 1, seq_len] -> [batch, seq_len, head_dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [batch, seq_len, head_dim]
            emb = torch.cat((freqs, freqs), dim = -1)
            # cos, sin: [batch, seq_len, head_dim]
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    # build the [-x2, x1, -x4, x3] tensor for the sin part of the PE
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 : ] # Takes the second half of the last dimension 
    return torch.cat((-x2, x1), dim = -1)
    

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim) # add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # add the head dimension
    # Apply the formula (34) of RoPE paper
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) # 1 / sqrt(...)
    
    def forward(self, x):
        output = self._norm(x.float())
        # llama: x.to(float16) * w
        # gemma: (x*w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # introduce a new dimension regarding the number of repititions and then reshape and return
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # we need layer_idx since each layer will have it's own KV-cache

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size # size of embedding
        self.num_heads = config.num_attention_heads # Q heads
        self.head_dim = config.head_dim # dimensions of each head
        self.num_key_value_heads = config.num_key_value_heads # KV heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # number of heads in each group
        self.max_position_embeddings = config.max_position_embeddings # how much positions we can encode in ROPE
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        # Create wq, wk, wv
        # hidden_size: 1024
        # heads = 8
        # KV heads = 1
        # head_dim = 1024 / 8 = 128
        # Wq: [1024, 8 * 128]
        # Wk: [1024, 1 * 128]
        # Wv: [1024, 1 * 128]
        # Wo: [8 * 128, 1024]
        # Multiple Query Attention: 1 KV head to many Q heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # ROPE
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() # [batch, seq_len, hidden_size]
        # [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads_q * head_dim]
        query_states = self.q_proj(hidden_states)
        # [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads_kv * head_dim]
        key_states = self.k_proj(hidden_states)
        # [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads_kv * head_dim]
        value_states = self.v_proj(hidden_states)
        # [batch, seq_lenm, num_heads_q * head_dim] -> [batch, seq_len, num_heads_q, head_dim] -> [batch, num_heads_q, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, seq_lenm, num_heads_kv * head_dim] -> [batch, seq_len, num_heads_kv, head_dim] -> [batch, num_heads_kv, seq_len, head_dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [batch, seq_lenm, num_heads_kv * head_dim] -> [batch, seq_len, num_heads_kv, head_dim] -> [batch, num_heads_kv, seq_len, head_dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        ### Apply ROPE
        # [batch, seq_len, head_dim], [batch, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len = None)
        # [batch, num_heads_q, seq_len, head_dim], [batch, num_heads_kv, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        key_states =  repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax
        # [batch, num_heads_q, seq_len_q, seq_len_kv]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p = self.attention_dropout, training=self.training)
        # Multiply by values. [batch, num_heads_q, seq_len_q, seq_len_kv] * [batch, num_heads_kv, seq_len_kv, head_dim] -> [batch, num_heads_q, seq_len_q, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output shoudl be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )
        
        # Transpose
        # [batch, num_heads_q, seq_len_q, head_dim] -> [batch, seq_len_q, num_heads_q, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concat all heads together
        # [batch, seq_len_q, num_heads_q head_dim] -> [batch, seq_len_q, num_heads_q * head_dim]
        attn_output = attn_output.view(bsz, q_len, -1)

        # Multiply by W_o
        # [batch, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # y = self.gate_proj(x) # [batch, seq_len, hidden_size] -> [batch, seq_len, intermediate_size]
        # y = torch.gelu(y, approximate="tanh") # [batch, seq_len, intermediate_layer]
        # j = self.up_proj(x) # [batch, seq_len, hidden_size] -> [batch, seq_len, intermediate_layer]
        # z = y * j [batch, seq_len, intermediate_size]
        # z = self.down_proj(z) # [batch, seq_len, intermediate_size] -> [batch, seq_len, hidden_size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


    
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config = config, layer_idx = layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        ### 1. Pre-Attention: x = attn(layernorm(x)) + x
        
        # [batch, seq_len, hidden_size]
        residue = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # [batch, seq_len, hidden_size]
        hidden_states, _ = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache,
        )

        # [batch, seq_len, hidden_size]
        hidden_states = residue + hidden_states

        ### 2. Post Attention: x = x + mlp(post_attn_layernorm(x))

        # [batch, seq_len, hidden_size]
        residue = hidden_states

        # [batch, seq_len, hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)

        # [batch, seq_len, hidden_size]
        hidden_states = self.mlp(hidden_states)

        # [batch, seq_len, hidden_size]
        hidden_states = residue + hidden_states

        return hidden_states


class PaliGemmaConfig():
    def __init__(
            self,
            vision_config = None, # config of SigLip Image Encoder
            text_config = None, # config of Gemma language model
            ignore_index = -100, # used during training
            image_token_index = 256000, # <image> token
            vocab_size = 257152, # vocab size of the model
            projection_dim = 2048, # hidden_size dimensions of the linear projection layer
            hidden_size = 2048, # embedding size of the language model
            pad_token_id = None, # not needed for now
            **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False # used by HF implementation
        self.pad_token_id = pad_token_id

        self.vision_config = SigLipVisionConfig(**vision_config) # Vision Encoder config
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id) # Text decoder config
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2 # num_patches
        self.vision_config.projection_dim = projection_dim

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [batch, seq_len, hidden_size]
        hidden_states = input_embeds
        # [batch, seq_len, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [batch, seq_len, hidden_size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache,
            )
        
        # [batch, seq_len, hidden_size]
        hidden_states = self.norm(hidden_states)

        return hidden_states
    


class GemmaForCausalLM(nn.Module):
    # Language model + Linear layer head
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    # Weight sharing between embedding layers and linear layer
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids : Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_embeds: [batch, seq_len, hidden_size]
        # outputs: [batch, seq_len, hidden_size]
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache,
        )

        # Apply linear head to generated embeddings
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits" : logits,
        }

        if kv_cache is not None:
            # Return the update cache as well
            return_data["kv_cache"] = kv_cache

        return return_data

        


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config : PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [batch, num_patches, embed_dim] -> [batch, num_patches, projection_dim] where projection_dim = hidden_size of text_config
        hidden_states = self.linear(image_features)
        return hidden_states
        

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config) # SigLip encoder
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config) # Linear projection layer
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config) # Transformer decoder
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        # weight sharing
        return self.language_model.tie_weights() # ties together output embeddings and linear layer in transformer decoder
    
    def _merge_input_ids_with_image_features(
            self, image_features: torch.Tensor, input_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache : Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        # 1. Scale the image features [batch, seq_len, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # 2. Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens
        # [batch, seq_len, embed_dim]
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        # [batch, seq_len] for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # [batch, seq_len] for image tokens
        image_mask = input_ids == self.config.image_token_index
        # [batch, seq_len] for padding tokens as well
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # Add the image embeddings
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        ### CREATE THE ATTENTION MASK ###
        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            # We create the mask in such a way that we dont want the Q to attend to future values
            # In PaliGemma, we do not mask the input prompt and image tokens!
            # As we want it to have access to the entire image and usually the prompt is very small and describes
            # the model what it needs to do so we keep it as it is.
            # Usually we mask the prompt as well but in this case, the authors chose not to! IMPORTANT POINT!!
            # Thus, causality is only in the generated tokens (but only for training)
            # But for PaliGemma, we do not mask out anything
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in tis case we don't need to mask anything, since each query should be able to attend all previous tokens
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )


        # Add the head dimensionL
        # [Batch, Q_len, KV_len] -> [Batch, Num_heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # Next, we generate the positions which will be used by ROPE
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position for the new generated token
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(
            self,
            input_ids: torch.LongTensor = None, # output from processing_paligemma which consists of image tokens and prompts
            pixel_values: torch.FloatTensor =  None, # images output from processing_paligemma
            attention_mask : Optional[torch.Tensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded" # since we are working with only 1 image at a time

        # 1. Extract the input embeddings: ids -> embeddings
        # [batch, seq_len, hidden_size]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and image tokens
        # 2.1 Get image tokens
        # [batch, channels, height, width] -> [batch, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(input_embeds.dtype))

        # 2.2 Resize them using a linear layer to match input embeddings for merging later
        # [batch, num_patches, embed_size] -> [batch, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # 2.3 Combine image_features with input embeddings which already have a placeholder space for image tokens (denoted by <image>)
        # image features: features extracted from siglip encoder
        # input_embeds: embeddings extracted from the language model
        # input_ids: ids from the language model
        input_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache)

        # 3. Pass the input_embeds with image info and prompt info to the language model
        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache,
        )

        return outputs


 
