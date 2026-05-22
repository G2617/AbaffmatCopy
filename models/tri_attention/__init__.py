# Copyright 2024 ByteDance and/or its affiliates.
# Copyright 2025 Shad Nygren, Virtual Hipster Corporation
#
# Modifications Copyright 2026 Eamon Byrne
# Modified for AbAffMat to support antibody-antigen scoring, yeast-display preference fine-tuning, and reproducible benchmarking.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by Shad Nygren, Virtual Hipster Corporation (2025):
# - Added fallback implementation for GPUs without Triton support
# - Enables compatibility with consumer GPUs (RTX 3090/4090)
# - Falls back to PyTorch's scaled_dot_product_attention when Triton unavailable

import warnings
from typing import Optional

import torch

try:
    import triton
    from models.tri_attention.op import TriAttention, TriAttentionFunction

    TRITON_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TRITON_AVAILABLE = False
    triton_error = str(e)

    class TriAttentionFunction:
        """Fallback implementation using PyTorch's native attention."""

        @staticmethod
        def apply(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            bias1: Optional[torch.Tensor] = None,
            bias2: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ) -> torch.Tensor:
            B, N, S, H, D = q.shape

            q = q.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
            k = k.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)
            v = v.permute(0, 1, 3, 2, 4).reshape(B * N * H, S, D)

            attn_mask = None
            if bias1 is not None or bias2 is not None:
                attn_mask = 0
                if bias1 is not None:
                    attn_mask = attn_mask + bias1.reshape(B * N * H, 1, S)
                if bias2 is not None:
                    attn_mask = attn_mask + bias2.reshape(B * N * H, S, 1)

            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
            ):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0 if deterministic else 0.0,
                    is_causal=False,
                )

            return out.reshape(B, N, H, S, D).permute(0, 1, 3, 2, 4)

    class TriAttention(torch.nn.Module):
        """Fallback TriAttention module using PyTorch."""

        def __init__(self) -> None:
            super().__init__()
            warnings.warn(
                f"Triton not available ({triton_error}). Using PyTorch fallback for attention. "
                "This may impact performance but maintains functionality.",
                UserWarning,
                stacklevel=2,
            )

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            bias1: Optional[torch.Tensor] = None,
            bias2: Optional[torch.Tensor] = None,
            deterministic: bool = False,
        ) -> torch.Tensor:
            return TriAttentionFunction.apply(q, k, v, bias1, bias2, deterministic)


__all__ = ["TriAttention", "TriAttentionFunction", "TRITON_AVAILABLE"]
