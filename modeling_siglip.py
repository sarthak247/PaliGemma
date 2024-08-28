from typing import Optional, Tuple
import torch
import torch.nn as nn

class SigLipVisionConfig:
    def __init__(
            self,
            hidden_size = 768, # size of the embedding vector
            intermediate_size = 3072, # size of FF layer
            num_hidden_layers = 12, # layers
            num_attention_heads = 12, # heads
            num_channels = 3, # RGB
            image_size = 224, # smallest image size of paligemma
            patch_size = 16, # 16x16 patches
            layer_norm_eps = 1e-6,
            attention_dropout = 0.0,
            num_image_tokens: int = None, # number of output image embeddings for each image
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
        self.patch_size = patch_size


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_size = config.hidden_size # embedding size
        self.image_size = config.image_size
        self.patch_size = config.patch_size # size of patch
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid', # no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches # number of embeddings to return
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [batch, channels, height, width]

        # convolve over our images to get patches of shape [batch, embed_dim, num_patches_h, num_patches_w]
        # since num_patches_h = num_patches_w we can also write [batch, embed_dim, num_patches]
        # here num_patches_h = height // patch_size
        patch_embeds = self.patch_embedding(pixel_values)

        # [batch, embed_dim, num_patches_h, num_patches_w] -> [batch, embed_dim, num_patches]
        # where num_patches = num_patches_h * num_patches_w
        embeddings = patch_embeds.flatten(2)

        # [batch, embed_size, num_patches] -> [batch, num_patches, embed_size]
        embeddings = embeddings.transpose(1, 2)

        # Add PE to each patch
        embeddings = embeddings + self.position_embeddings(self.position_ids)

        # [batch, num_patches, embed_size]
        return embeddings


class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [batch, num_patches, embed_size]
        residual = hidden_states

        # [batch, num_patches, embed_size] -> [batch, num_patches, embed_size]
        hidden_states = self.layer_norm1(hidden_states)

        # [batch, num_patches, embed_size] -> [batch, num_patches, embed_size]
        hidden_states, _ = self.self_attn(hidden_states =  hidden_states)

        # [batch, num_patches, embed_size]
        hidden_states = residual + hidden_states

        # [batch, num_patches, embed_size]
        residual = hidden_states

        # [batch, num_patches, embed_size] -> [batch, num_patches, embed_size]
        hidden_states = self.layer_norm2(hidden_states)

        # [batch, num_patches, embed_size] -> [batch, num_patches, embed_size]
        hidden_states = self.mlp(hidden_states)

        # [batch, num_patches, embed_size]
        hidden_states = residual + hidden_states

        # [batch, num_patches, embed_size]
        return hidden_states
    

class SigLipMLP(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch, num_patches, embed_size] -> [batch, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)

        # [batch_size, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        # [batch, num_patches, intermediate_size] -> [batch, num_patches, embed_size]
        hidden_states = self.fc2(hidden_states)

        # [batch, num_patches, embed_size]
        return hidden_states

  
class SigLipAttention(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # 1/sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
            self, 
            hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [batch, num_patches, embed_dim]
        batch_size, seq_len, _ = hidden_states.size()

        # query_states: [batch, num_patches, embed_size]
        query_states = self.q_proj(hidden_states)

        # key_states: [batch, num_patches, embed_size]
        key_states = self.k_proj(hidden_states)

        # value_states: [batch, num_patches, embed_size]
        value_states = self.v_proj(hidden_states)

        # [batch, num_patches, embed_size] -> [batch, num_patches, n_heads, head_dim] -> [batch, n_heads, num_patches, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention using formula Q*K.T/sqrt(d_k)
        # [batch, n_heads, num_patches, head_dim] * [batch, n_heads, head_dim, num_patches] -> [batch, n_heads, num_patches, num_patches]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is {attn_weights.size()}"
            )
        
        # Apply softmax row-wise. attn_weights: [batch, n_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype=torch.float32).to(query_states.dtype)

        # Apply dropout during training
        attn_weights = nn.functional.dropout(attn_weights, p = self.dropout, training=self.training)

        # Multiply by V... attn_weights: [batch, n_heads, num_patches, head_dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size: {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}"
            )
        
        # [batch, n_heads, num_patches, head_dim] -> [batch, num_patches, n_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # [batch, num_patches, n_heads, head_dim] -> [batch, num_patches, embed_size]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)


        # attn_output: [batch, num_patches, embed_size]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SigLipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
            self,
            input_embeds: torch.Tensor
    ) -> torch.Tensor:
        # input_embeds: [batch, num_patches, embed_size]
        hidden_states = input_embeds

        for layer in self.layers:
            # [batch, num_patches, embed_size] -> [batch, num_patches, embed_size]
            hidden_states = layer(hidden_states)

        # [batch, num_patches, embed_size]
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size # size of the embedding vector

        self.embeddings = SigLipVisionEmbeddings(config) # extract embeddings
        self.encoder = SigLipEncoder(config) # encode the embeddings
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps) # layernorm

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch, channels, height, width] -> [batch, patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(input_embeds = hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [batch, channels, height, width] -> (batch, num_patches, embed_dim) where num_patches = num_image_tokens
        return self.vision_model(pixel_values = pixel_values)
