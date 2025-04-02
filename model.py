from collections import OrderedDict
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

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
    """

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
        in_channels: int = 4,  # 5 channels: drums, bass, vocals, other + beat frames
        dropout: dict = {"frontend": 0.1, "transformer": 0.2},
    ):
        super().__init__()
        
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
                    head_dim,
                    rotary_embed,
                    dropout["frontend"],
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
        
        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=n_heads,
                dim_feedforward=transformer_dim * ff_mult,
                dropout=dropout["transformer"],
                activation="gelu",
                batch_first=True
            ),
            num_layers=n_layers,
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
    def make_stem(spect_dim: int, stem_dim: int, in_channels: int = 5) -> nn.Module:
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
        head_dim: int,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ) -> nn.Module:
        """
        Creates a frontend block with convolution and attention.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            head_dim (int): Dimension of each attention head.
            rotary_embed (RotaryEmbedding): Rotary embedding for attention.
            dropout (float): Dropout rate.

        Returns:
            nn.Module: Frontend block.
        """
        return nn.Sequential(
            # Convolution block
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=(2, 3),
                stride=(2, 1),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input spectrograms of shape (batch_size, channels=4, frequency, time).
                            Each channel represents a different spectrogram or beat frame.

        Returns:
            torch.Tensor: Predicted brightness values with shape (batch_size, time_steps, 1)
        """
        # Process through frontend CNN blocks
        x = self.frontend(x)  # Output: [batch, channels, freq, time]
        
        # Permute to get [batch, time, channels, freq]
        batch_size, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        
        # Reshape to [batch, time, channels*freq] for the linear projection
        x = x.reshape(batch_size, time, channels * freq)
        
        # Project to transformer dimension
        x = self.projection(x)  # [batch, time, transformer_dim]
        
        # Pass through transformer
        transformer_output = self.transformer_blocks(x)  # [batch, time, transformer_dim]
        
        # Apply holistic attention to highlight important features across all tracks
        attention_weights = self.attention(transformer_output)
        attended_output = transformer_output * attention_weights
        
        # Generate brightness prediction
        brightness = torch.sigmoid(self.brightness_head(attended_output))  # [batch, time, 1]
        
        return brightness