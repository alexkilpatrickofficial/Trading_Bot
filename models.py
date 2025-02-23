import torch
import torch.nn as nn
import logging
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from chromosome import BLOCK_TYPES, MAX_BLOCKS  # Local constants
from config import FEATURE_COLUMNS, TOTAL_FEATURE_COUNT  # Must match your environment config

logger = logging.getLogger(__name__)

class ModularBlock(nn.Module):
    """
    A modular block that implements one of several layer types: CNN, GRU, LSTM,
    TCN, TRANSFORMER, MLP, or NONE.
    """
    def __init__(self, block_type, in_dim, out_dim, kernel_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.block_type = BLOCK_TYPES.get(block_type, "NONE")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = max(0.0, min(1.0, dropout))

        if self.block_type == "CNN":
            self.layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
        elif self.block_type == "TCN":
            layers = []
            current_channels = in_dim
            for _ in range(num_layers):
                layers.append(nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    dilation=1
                ))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                current_channels = out_dim
            self.layer = nn.Sequential(*layers)
        elif self.block_type == "GRU":
            self.layer = nn.GRU(
                input_size=in_dim,
                hidden_size=out_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=(self.dropout if num_layers > 1 else 0.0)
            )
        elif self.block_type == "LSTM":
            self.layer = nn.LSTM(
                input_size=in_dim,
                hidden_size=out_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=(self.dropout if num_layers > 1 else 0.0)
            )
        elif self.block_type == "TRANSFORMER":
            if self.in_dim % self.num_heads != 0:
                logger.warning(
                    f"Transformer block: in_dim ({self.in_dim}) not divisible by num_heads ({self.num_heads}). Setting num_heads to 1."
                )
                self.num_heads = 1
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.in_dim,
                nhead=self.num_heads,
                dim_feedforward=self.in_dim * 4,
                dropout=self.dropout,
                batch_first=True
            )
            self.layer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        elif self.block_type == "MLP":
            layers = []
            current_size = in_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(current_size, out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
                current_size = out_dim
            self.layer = nn.Sequential(*layers)
        else:
            self.layer = None

    def forward(self, x):
        if self.block_type == "NONE" or self.layer is None:
            return x

        if self.block_type in ["CNN", "TCN"]:
            # Expect input shape [B, T, C] and convert to [B, C, T]
            x = x.permute(0, 2, 1)
            out = self.layer(x)
            # Convert back to [B, T, C]
            out = out.permute(0, 2, 1)
            # Debug assertion
            assert out.shape[-1] == self.out_dim, f"Expected output dim {self.out_dim}, got {out.shape[-1]}"
            return out

        elif self.block_type in ["GRU", "LSTM"]:
            out, _ = self.layer(x)
            # Check last dimension equals out_dim
            assert out.shape[-1] == self.out_dim, f"Expected output dim {self.out_dim}, got {out.shape[-1]}"
            return out

        elif self.block_type == "TRANSFORMER":
            out = self.layer(x)
            # For transformer, input and output dims remain the same.
            return out

        elif self.block_type == "MLP":
            B, T, F = x.size()
            out = self.layer(x.reshape(B * T, F))
            out = out.reshape(B, T, self.out_dim)
            # Check shape consistency
            assert out.shape[-1] == self.out_dim, f"MLP block: expected {self.out_dim}, got {out.shape[-1]}"
            return out

        return x

def simulate_modular_blocks(chromosome, window_size, input_dim, device="cuda"):
    """
    Simulate a forward pass through the modular blocks to determine
    the final output channel count.
    """
    num_blocks = int(chromosome[3])
    current_dim = input_dim
    block_start = 4
    x = torch.zeros(1, window_size, input_dim, device=device)
    logger.debug(f"[simulate_modular_blocks] Initial dummy input shape: {x.shape}")

    for i in range(num_blocks):
        idx = block_start + i * 5
        b_type = int(chromosome[idx])
        out_dim = int(chromosome[idx + 1])
        k_size = int(chromosome[idx + 2])
        n_heads = int(chromosome[idx + 3])
        n_layers = int(chromosome[idx + 4])

        block = ModularBlock(
            block_type=b_type,
            in_dim=current_dim,
            out_dim=out_dim,
            kernel_size=k_size,
            num_heads=n_heads,
            num_layers=n_layers,
            dropout=0.2  # For simulation purposes
        ).to(device)

        try:
            x = block(x)
            logger.debug(f"[simulate_modular_blocks] After block {i}, shape: {x.shape}")
        except Exception as e:
            logger.error(f"Error simulating block {i}: {e}")
            return None, current_dim

        if b_type != 6:  # 6 corresponds to "NONE"
            current_dim = out_dim

    return x.size(-1), current_dim

class DynamicHybridFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that applies a series of modular blocks, a compression transformer,
    an output transformer, temporal pooling, and a final projection to a latent dimension (256).
    Also provides an auxiliary head for price prediction.
    Returns a dict with:
      "features": [B, 256] for RL policy/value,
      "price_prediction": [B, 1].
    """
    def __init__(self, observation_space, chromosome, features_dim=256, device="cuda", comp_transformer_params=None):
        total_features = TOTAL_FEATURE_COUNT
        self.window_size = observation_space.shape[0] // total_features
        self.input_dim = total_features

        super().__init__(observation_space, features_dim)
        self.device = torch.device(device)
        self.original_shape = (self.window_size, self.input_dim)

        logger.info(f"Initializing DynamicHybridFeatureExtractor with latent dimension: {features_dim}")
        logger.debug(f"[FeatureExtractor] Original observation space shape: {observation_space.shape}")
        logger.debug(f"[FeatureExtractor] Expected per-timestep features: {self.input_dim}, window size: {self.window_size}")

        # Set dropout from chromosome (no clamping here, just use value)
        self.dropout = float(chromosome[4 + MAX_BLOCKS * 5 + 2])
        self.dropout = max(0.0, min(1.0, self.dropout))
        logger.debug(f"[FeatureExtractor] Using dropout: {self.dropout}")

        # Build modular blocks
        self.num_blocks = int(chromosome[3])
        self.blocks = nn.ModuleList()
        current_dim = self.input_dim
        block_start = 4
        for i in range(self.num_blocks):
            idx = block_start + i * 5
            b_type = int(chromosome[idx])
            out_dim = int(chromosome[idx + 1])
            k_size = int(chromosome[idx + 2])
            n_heads = int(chromosome[idx + 3])
            n_layers = int(chromosome[idx + 4])
            block = ModularBlock(
                block_type=b_type,
                in_dim=current_dim,
                out_dim=out_dim,
                kernel_size=k_size,
                num_heads=n_heads,
                num_layers=n_layers,
                dropout=self.dropout
            )
            self.blocks.append(block)
            logger.debug(f"[FeatureExtractor] Added block {i} of type {BLOCK_TYPES.get(b_type, 'NONE')} with in_dim {current_dim} and out_dim {out_dim}")
            if b_type != 6:
                current_dim = out_dim

        # Simulate modular blocks to determine output dimension
        simulated_dim, simulated_current_dim = simulate_modular_blocks(chromosome, self.window_size, self.input_dim, device=self.device)
        if simulated_dim is None:
            raise ValueError("Simulation of modular blocks failed.")
        if simulated_current_dim != current_dim:
            logger.warning(f"[FeatureExtractor] Discrepancy in modular block output: computed {current_dim} vs simulated {simulated_current_dim}. Using simulated value.")
        current_dim = simulated_dim
        logger.info(f"[FeatureExtractor] Modular blocks output dimension: {current_dim}")

        # Compression transformer parameters from chromosome tail
        comp_start = 4 + MAX_BLOCKS * 5 + 3
        comp_num_layers = int(chromosome[comp_start])
        comp_nhead = int(chromosome[comp_start + 1])
        comp_dropout = float(chromosome[comp_start + 2])
        comp_dropout = max(0.0, min(1.0, comp_dropout))
        original_comp_nhead = comp_nhead
        while comp_nhead > 1 and current_dim % comp_nhead != 0:
            comp_nhead -= 1
        if comp_nhead < 1:
            comp_nhead = 1
        if comp_nhead != original_comp_nhead:
            logger.warning(f"[FeatureExtractor] Adjusted compression transformer nhead from {original_comp_nhead} to {comp_nhead} for current_dim {current_dim}")
        logger.info(f"[FeatureExtractor] Compression transformer params: num_layers={comp_num_layers}, nhead={comp_nhead}, dropout={comp_dropout}")
        self.comp_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=current_dim,
                nhead=comp_nhead,
                dim_feedforward=current_dim * 2,
                dropout=comp_dropout,
                batch_first=True
            ),
            num_layers=comp_num_layers
        )

        # Output transformer parameters from chromosome tail
        out_start = comp_start + 3
        out_num_layers = int(chromosome[out_start])
        out_nhead = int(chromosome[out_start + 1])
        out_hidden = int(chromosome[out_start + 2])
        original_out_nhead = out_nhead
        while out_nhead > 1 and out_hidden % out_nhead != 0:
            out_nhead -= 1
        if out_nhead < 1:
            out_nhead = 1
        if out_nhead != original_out_nhead:
            logger.warning(f"[FeatureExtractor] Adjusted output transformer nhead from {original_out_nhead} to {out_nhead} for out_hidden {out_hidden}")
        logger.info(f"[FeatureExtractor] Output transformer params: num_layers={out_num_layers}, nhead={out_nhead}, hidden={out_hidden}")

        self.proj_out = nn.Linear(current_dim, out_hidden) if out_hidden != current_dim else nn.Identity()
        self.out_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_hidden,
                nhead=out_nhead,
                dim_feedforward=out_hidden * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=out_num_layers
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Final projection to latent dimension (fixed at 256)
        self.latent_dim = 256
        logger.info(f"[FeatureExtractor] Using latent dimension: {self.latent_dim}")
        self.fc_out = nn.Sequential(
            nn.Linear(out_hidden, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        logger.debug(f"[FeatureExtractor] fc_out output dimension (expected latent dim): {self.latent_dim}")

        # Auxiliary price predictor head
        self.price_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),  # 256 -> 128
            nn.ReLU(),
            nn.Linear(self.latent_dim // 2, 1)                  # 128 -> 1
        )
        logger.info(f"[FeatureExtractor] Price predictor layer 1 weight shape: {self.price_predictor[0].weight.shape}")
        logger.info(f"[FeatureExtractor] Price predictor layer 2 weight shape: {self.price_predictor[2].weight.shape}")

        self.to(self.device)

    def forward(self, observations: torch.Tensor) -> dict:
        """
        Process flattened observations and return:
          {
            "features": [B, 256],
            "price_prediction": [B, 1]
          }
        """
        B = observations.shape[0]
        # Reshape the flattened observation back to [B, window_size, input_dim]
        x = observations.reshape(B, self.window_size, self.input_dim).to(self.device)
        logger.debug(f"[FeatureExtractor] Reshaped input: {x.shape}")

        # Pass through modular blocks with debug logging at each block.
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            logger.debug(f"[FeatureExtractor] After block {block_idx}, shape: {x.shape}")

        # Compression transformer stage
        x = self.comp_transformer(x)
        logger.debug(f"[FeatureExtractor] After comp_transformer, shape: {x.shape}")

        # Project to output transformer dimension and pass through it
        x = self.proj_out(x)
        x = self.out_transformer(x)
        logger.debug(f"[FeatureExtractor] After out_transformer, shape: {x.shape}")

        # Temporal pooling over time dimension
        x = x.permute(0, 2, 1)  # Change to [B, hidden, T]
        x = self.temporal_pool(x).squeeze(-1)  # Now [B, hidden]
        logger.debug(f"[FeatureExtractor] After temporal_pool, shape: {x.shape}")

        # Final projection to latent features
        features = self.fc_out(x)
        assert features.shape[-1] == self.latent_dim, f"Expected latent dim {self.latent_dim}, got {features.shape[-1]}"
        logger.debug(f"[FeatureExtractor] fc_out features shape: {features.shape}")

        # Price prediction head
        price_prediction = self.price_predictor(features)
        logger.debug(f"[FeatureExtractor] price_prediction shape: {price_prediction.shape}")

        # For extra debugging, you can log a summary of the columns/order if needed.
        # (This would require that you pass in the column order externally.)
        return {
            "features": features,           # Shape [B, 256]
            "price_prediction": price_prediction  # Shape [B, 1]
        }
