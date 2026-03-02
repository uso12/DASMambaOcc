import logging
import warnings
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


LOGGER = logging.getLogger(__name__)


class _FallbackMixerBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MambaRefinementSubHead(nn.Module):
    """OccMamba-inspired refinement head applied after coarse occupancy logits."""

    def __init__(
        self,
        bev_channels: int,
        num_classes: int,
        dz: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        residual_scale: float = 0.35,
        scan_orders: Sequence[str] = ("xy", "yx"),
        use_mamba: bool = True,
        strict_mamba: bool = False,
        dropout: float = 0.0,
        lcp_kernel: int = 5,
        use_checkpoint: bool = False,
        checkpoint_min_tokens: int = 20000,
        warn_long_sequence: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if lcp_kernel < 1 or lcp_kernel % 2 == 0:
            raise ValueError("lcp_kernel must be a positive odd integer")
        if not scan_orders:
            raise ValueError("scan_orders must contain at least one order")
        unsupported = [order for order in scan_orders if order not in ("xy", "yx")]
        if unsupported:
            raise ValueError(f"Unsupported scan_orders entries: {unsupported}")

        self.num_classes = num_classes
        self.dz = dz
        self.residual_scale = residual_scale
        self.scan_orders = tuple(scan_orders)
        self.requested_mamba = bool(use_mamba)
        self.use_checkpoint = bool(use_checkpoint)
        self.checkpoint_min_tokens = int(checkpoint_min_tokens)
        self.warn_long_sequence = bool(warn_long_sequence)
        self._warned_long_sequence = False

        token_dim = bev_channels + num_classes * dz
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.use_mamba = False
        self.mixer_layers = nn.ModuleList()
        self.mixer_norms = nn.ModuleList()

        if use_mamba:
            try:
                from mamba_ssm import Mamba  # type: ignore

                self.mixer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
                self.mixer_layers = nn.ModuleList(
                    [Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)]
                )
                self.use_mamba = True
            except Exception as exc:
                self.use_mamba = False
                if strict_mamba:
                    raise ImportError(
                        "MambaRefinementSubHead was configured with use_mamba=True, "
                        "but mamba_ssm is unavailable."
                    ) from exc
                warnings.warn(
                    "mamba_ssm is unavailable; falling back to MLP mixer refinement.",
                    stacklevel=2,
                )
                LOGGER.warning("Falling back to _FallbackMixerBlock because mamba_ssm import failed: %s", exc)

        if not self.use_mamba:
            self.mixer_layers = nn.ModuleList(
                [_FallbackMixerBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
            )

        self.lcp = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=lcp_kernel,
                stride=1,
                padding=lcp_kernel // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.out_proj = nn.Linear(hidden_dim, num_classes * dz)

    @staticmethod
    def _make_xy_pos(batch_size: int, dx: int, dy: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        xs = torch.linspace(-1.0, 1.0, dx, device=device, dtype=dtype).view(dx, 1).expand(dx, dy)
        ys = torch.linspace(-1.0, 1.0, dy, device=device, dtype=dtype).view(1, dy).expand(dx, dy)
        pos = torch.stack([xs, ys], dim=-1)
        return pos.unsqueeze(0).expand(batch_size, dx, dy, 2)

    def _run_mixer(self, seq: torch.Tensor) -> torch.Tensor:
        can_checkpoint = self.use_checkpoint and self.training and seq.requires_grad

        if self.use_mamba:
            for norm, layer in zip(self.mixer_norms, self.mixer_layers):
                if can_checkpoint:
                    def _mamba_block(x, _norm=norm, _layer=layer):
                        return _layer(_norm(x))
                    seq = seq + checkpoint(_mamba_block, seq)
                else:
                    seq = seq + layer(norm(seq))
            return seq

        for layer in self.mixer_layers:
            if can_checkpoint:
                def _fallback_block(x, _layer=layer):
                    return _layer(x)
                seq = checkpoint(_fallback_block, seq)
            else:
                seq = layer(seq)
        return seq

    def _scan_once(self, tokens: torch.Tensor, order: str) -> torch.Tensor:
        # tokens: [B, Dx, Dy, C]
        b, dx, dy, c = tokens.shape
        if order == "xy":
            seq = tokens.reshape(b, dx * dy, c)
            seq = self._run_mixer(seq)
            return seq.view(b, dx, dy, c)

        if order == "yx":
            seq = tokens.permute(0, 2, 1, 3).reshape(b, dx * dy, c)
            seq = self._run_mixer(seq)
            return seq.view(b, dy, dx, c).permute(0, 2, 1, 3)

        raise ValueError(f"Unsupported scan order: {order}")

    def forward(self, bev_feat_xy: torch.Tensor, occ_logits: torch.Tensor) -> torch.Tensor:
        # bev_feat_xy: [B, C_bev, Dy, Dx]
        # occ_logits: [B, Dx, Dy, Dz, C_cls]
        b, dx, dy, dz, c_cls = occ_logits.shape
        if c_cls != self.num_classes or dz != self.dz:
            raise ValueError(
                f"Unexpected occupancy shape: got Dz={dz}, C={c_cls}, expected Dz={self.dz}, C={self.num_classes}"
            )
        seq_len = dx * dy
        if (
            self.warn_long_sequence
            and self.training
            and not self.use_checkpoint
            and seq_len >= self.checkpoint_min_tokens
            and not self._warned_long_sequence
        ):
            warnings.warn(
                f"Mamba refinement sequence length is {seq_len}. "
                "Enable use_checkpoint=True or reduce scan complexity to lower VRAM pressure.",
                stacklevel=2,
            )
            self._warned_long_sequence = True

        bev_tokens = bev_feat_xy.permute(0, 3, 2, 1).contiguous()  # [B, Dx, Dy, C_bev]
        occ_tokens = occ_logits.reshape(b, dx, dy, dz * c_cls)

        tokens = torch.cat([bev_tokens, occ_tokens], dim=-1)
        tokens = self.token_proj(tokens)

        pos = self._make_xy_pos(b, dx, dy, tokens.device, tokens.dtype)
        tokens = tokens + self.pos_proj(pos)

        scanned = [self._scan_once(tokens, order) for order in self.scan_orders]
        fused = torch.stack(scanned, dim=0).mean(dim=0)

        # Local context processor (OccMamba LCP style) over BEV grid.
        fused_2d = fused.permute(0, 3, 2, 1).contiguous()  # [B, C, Dy, Dx]
        fused_2d = fused_2d + self.lcp(fused_2d)
        fused = fused_2d.permute(0, 3, 2, 1).contiguous()

        residual = self.out_proj(fused).view(b, dx, dy, dz, c_cls)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.nan_to_num(occ_logits + self.residual_scale * residual, nan=0.0, posinf=1e4, neginf=-1e4)
