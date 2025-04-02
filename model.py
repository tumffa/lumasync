from collections import OrderedDict
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
import roformer

class LightingModel(nn.Module):
    """
    A neural network model for predicting lighting brightness values from audio spectrograms.
    Architecture based on beat tracking model BeatThis: https://arxiv.org/abs/2407.21658

    Args:
        spect_dim (int): The dimension of the input spectrogram (default: 128).
        transformer_dim (int): The dimension of the main transformer blocks (default: 512).
        ff_mult (int): The multiplier for the feed-forward dimension in the transformer blocks (default: 4).
        n_layers (int): The number of transformer blocks (default: 6).
        head_dim (int): The dimension of each attention head for the transformers (default: 32).
        stem_dim (int): The output dimension of the stem convolutional layer (default: 32).
        in_channels (int): Number of input spectrogram channels (default: 5).
        dropout (dict): A dictionary specifying the dropout rates for different parts of the model
            (default: {"frontend": 0.1, "transformer": 0.2}).
        max_time_dim (int): Maximum time dimension after downsampling (default: 512).
            Set to None to disable time dimension downsampling.
        transformer_chunk_size (int): Size of chunks for transformer processing (default: 2048).
            Set to None to disable transformer chunking.
    """

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
        in_channels: int = 1,
        dropout: dict = {"frontend": 0.1, "transformer": 0.2},
        max_time_dim=None,
        transformer_chunk_size=None
    ):
        super().__init__()
        
        # Store chunking parameters
        self.max_time_dim = max_time_dim
        self.transformer_chunk_size = transformer_chunk_size
        
        # Shared rotary embedding for transformer blocks
        rotary_embed = RotaryEmbedding(head_dim)

        # Create the frontend
        # - Stem
        stem = self.make_stem(spect_dim, stem_dim, in_channels)
        spect_dim //= 4  # Frequencies are reduced by a factor of 4 due to convolution
        
        # - Frontend blocks
        frontend_blocks = []
        dim = stem_dim
        for _ in range(3):
            frontend_blocks.append(
                self.make_frontend_block(
                    dim,
                    dim * 2,
                    partial_transformers=False,
                    head_dim=head_dim,
                    rotary_embed=rotary_embed,
                    dropout=dropout["frontend"],
                )
            )
            dim *= 2
            spect_dim //= 2  # Frequencies are reduced by a factor of 2 per block
        
        # Combine stem and frontend blocks
        self.frontend = nn.Sequential(
            OrderedDict(stem=stem, blocks=nn.Sequential(*frontend_blocks))
        )
        
        # Add a separate module for the flattening and projection
        self.projection = nn.Linear(dim * spect_dim, transformer_dim)

        # Create the transformer blocks
        assert (
            transformer_dim % head_dim == 0
        ), "transformer_dim must be divisible by head_dim"
        n_heads = transformer_dim // head_dim
        
        self.transformer_blocks = roformer.Transformer(
            dim=transformer_dim,
            depth=n_layers,
            dim_head=head_dim,
            heads=n_heads,
            attn_dropout=dropout["transformer"],
            ff_dropout=dropout["transformer"],
            rotary_embed=rotary_embed,
            ff_mult=ff_mult,
            norm_output=True,
        )

        # Create a holistic attention layer to focus on important patterns
        self.attention = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.GELU(),
            nn.Linear(transformer_dim // 2, transformer_dim),
            nn.Sigmoid()
        )

        # Create the output head for brightness prediction
        self.brightness_head = nn.Linear(transformer_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def make_stem(spect_dim: int, stem_dim: int, in_channels: int = 1) -> nn.Module:
        """
        Creates the stem convolutional layer.

        Args:
            spect_dim (int): Input spectrogram dimension.
            stem_dim (int): Output dimension of the stem.
            in_channels (int): Number of input channels (spectrograms).

        Returns:
            nn.Module: Stem layer.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_dim,
                kernel_size=(4, 3),
                stride=(4, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )

    @staticmethod
    def make_frontend_block(
        in_dim: int,
        out_dim: int,
        partial_transformers: bool = True,
        head_dim: int | None = 32,
        rotary_embed: RotaryEmbedding | None = None,
        dropout: float = 0.1,
    ) -> nn.Module:
        """
        Creates a frontend block with optional partial transformers and convolution.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            partial_transformers (bool): Whether to use partial transformers.
            head_dim (int | None): Dimension of each attention head (required if partial_transformers is True).
            rotary_embed (RotaryEmbedding | None): Rotary embedding for attention (required if partial_transformers is True).
            dropout (float): Dropout rate.

        Returns:
            nn.Module: Frontend block.
        """
        if partial_transformers and (head_dim is None or rotary_embed is None):
            raise ValueError(
                "Must specify head_dim and rotary_embed for using partial_transformers"
            )
        return nn.Sequential(
            OrderedDict(
                partial=(
                    PartialFTTransformer(
                        dim=in_dim,
                        dim_head=head_dim,
                        n_head=in_dim // head_dim,
                        rotary_embed=rotary_embed,
                        dropout=dropout,
                    )
                    if partial_transformers
                    else nn.Identity()
                ),
                conv2d=nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=(2, 3),
                    stride=(2, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                norm=nn.BatchNorm2d(out_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def _init_weights(module: nn.Module):
        """
        Initializes weights for the model.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass with configurable downsampling and chunking for sequence processing.
        
        Args:
            x: Input tensor of shape [batch, channels, freq, time]
            
        Returns:
            brightness: Output tensor of shape [batch, time, 1]
        """
        batch_size = x.shape[0]
        original_time = x.shape[-1]  # Store original time dimension
        
        # Process through frontend CNN blocks
        x = self.frontend(x)  # Output: [batch, channels, freq, time]
        
        # Get dimensions
        batch_size, channels, freq, time = x.shape
        
        # Apply time dimension downsampling if configured
        if self.max_time_dim is not None and time > self.max_time_dim:
            # Use a safer multi-stage approach rather than one big adaptive_avg_pool2d
            
            # Calculate how many stages we need (at most halve the dimension in each stage)
            # to avoid excessive shared memory usage
            current_time = time
            while current_time > self.max_time_dim:
                # Downsample by at most 50% in each stage
                target_time = max(self.max_time_dim, current_time // 2)
                
                # Use max pooling first to reduce memory pressure
                if current_time > 2 * self.max_time_dim:
                    kernel_size = (1, 2)
                    x = nn.functional.max_pool2d(x, kernel_size)
                    current_time = x.shape[-1]
                
                # Then use a small adaptive_avg_pool2d step
                x = nn.functional.adaptive_avg_pool2d(x, (freq, target_time))
                current_time = target_time
                
                # Force memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            time = current_time
        
        # Process the sequence normally 
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        x = x.reshape(batch_size, time, channels * freq)
        
        # Project to transformer dimension
        x = self.projection(x)  # [batch, time, transformer_dim]
        
        # Process through transformer in chunks if configured
        if self.transformer_chunk_size is None or time <= self.transformer_chunk_size:
            # Process whole sequence at once
            transformer_output = self.transformer_blocks(x)
        else:
            # Process in chunks
            chunk_size = self.transformer_chunk_size
            num_chunks = (time + chunk_size - 1) // chunk_size
            transformer_outputs = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, time)
                
                # Process this chunk
                chunk = x[:, start_idx:end_idx, :]
                chunk_output = self.transformer_blocks(chunk)
                transformer_outputs.append(chunk_output)
                
                # Force memory cleanup after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Concatenate chunks
            transformer_output = torch.cat(transformer_outputs, dim=1)
        
        # Force GPU memory cleanup after transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final steps
        attention_weights = self.attention(transformer_output)
        attended_output = transformer_output * attention_weights
        brightness = torch.sigmoid(self.brightness_head(attended_output))
        
        # Store the downsampling factor for potential upsampling
        self.last_downsample_factor = original_time / time
        
        return brightness

class PartialFTTransformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block once across frequencies and once across time. Same
    as applying two PartialRoformer() in sequence, but encapsulated in a single
    module. Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        # frequency directed partial transformer
        self.attnF = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffF = roformer.FeedForward(dim, dropout=dropout)
        # time directed partial transformer
        self.attnT = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffT = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        # frequency directed partial transformer
        x = rearrange(x, "b c f t -> (b t) f c")
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        # time directed partial transformer
        x = rearrange(x, "(b t) f c ->(b f) t c", b=b)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        x = rearrange(x, "(b f) t c -> b c f t", b=b)
        return x
