# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

from typing import Optional, Union

import torch
import torch.nn as nn

from models.modules.pairformer import PairformerStack
from models.modules.primitives import LinearNoBias
from openfold_local.model.primitives import LayerNorm


def resolve_scoring_head_class(head_name: str) -> type[nn.Module]:
    normalized = head_name.strip().lower().replace("-", "_")
    registry = {
        "gated_pooling": GatedPoolingHead,
        "gatedpooling": GatedPoolingHead,
        "affinity_head": GatedPoolingHead,
        "affinity": GatedPoolingHead,
        "pairformer_refine": PairformerRefinementHead,
        "pairformerrefine": PairformerRefinementHead,
        "pairformer_refinement": PairformerRefinementHead,
        "binder_head": PairformerRefinementHead,
        "binder": PairformerRefinementHead,
    }
    if normalized not in registry:
        supported = ["gated_pooling", "pairformer_refine"]
        raise ValueError(
            f"Unsupported scoring head '{head_name}'. Supported values: {supported}"
        )
    return registry[normalized]


class GatedPoolingHead(torch.nn.Module):
    """Gated pooling scoring head (formerly `AffinityHead`)."""

    def __init__(self, c_s: int, c_z: int, **kwargs):
        """
        Formerly `AffinityHead`.

        Args:
            c_s:
                Input channel dimension
        """
        super().__init__()

        self.c_s = c_s
        self.gate_ln = torch.nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        self.gate_linear = torch.nn.Linear(self.c_s, 1, bias=True)
        self.ln = torch.nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        # self.linear = torch.nn.Linear(self.c_s, 1)
        hidden_size = 512
        self.mlp = nn.Sequential(
            nn.Linear(self.c_s, hidden_size),  # First layer
            nn.ReLU(),                          # Activation
            nn.Linear(hidden_size, hidden_size), # Second layer
            nn.ReLU(),                          # Activation
            nn.Dropout(0.2),                    # Dropout for regularization
            nn.Linear(hidden_size, 1) # Output layer
        )

        self.c_z = c_z
        self.z_gate_ln = torch.nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_gate_linear = torch.nn.Linear(self.c_z, 1, bias=True)
        self.z_ln = torch.nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_mlp = nn.Sequential(
            nn.Linear(self.c_z, hidden_size),  # First layer
            nn.ReLU(),                          # Activation
            nn.Linear(hidden_size, hidden_size), # Second layer
            nn.ReLU(),                          # Activation
            nn.Dropout(0.2),                    # Dropout for regularization
            nn.Linear(hidden_size, 1) # Output layer
        )

    def forward(self, s_inputs, s, z, single_mask, pair_mask, **kwargs):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, 1] affinity prediction
        """
        # Learn a per-token importance weight and token-level score, then pool over tokens.
        gate = self.gate_linear(self.gate_ln(s)).sigmoid() * single_mask.unsqueeze(-1)
        s_affinity_logits = self.mlp(self.ln(s)) * single_mask.unsqueeze(-1)
        s_affinity = (gate * s_affinity_logits).sum(dim=-2)

        # Do the same for pairwise features, masking out padded token-token interactions.
        z_gate = (self.z_gate_linear(self.z_gate_ln(z))).sigmoid() * pair_mask.unsqueeze(-1)
        z_affinity_logits = self.z_mlp(self.z_ln(z)) * pair_mask.unsqueeze(-1)
        z_affinity = (z_gate * z_affinity_logits).mean(dim=-2).sum(dim=-2)
        
        # Final affinity score combines single-token and pairwise evidence.
        affinity = s_affinity + z_affinity
        return affinity


class PairformerRefinementHead(nn.Module):
    """Pairformer refinement scoring head (formerly `BinderHead`)."""

    def __init__(
        self,
        n_blocks: int = 2,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        max_atoms_per_token: int = 20,
        pairformer_dropout: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        stop_gradient: bool = True,
    ) -> None:
        """
        Formerly `BinderHead`.

        Args:
            n_blocks (int, optional): number of blocks for ConfidenceHead. Defaults to 4.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s_inputs (int, optional): hidden dim [for single embedding from InputFeatureEmbedder]. Defaults to 449.
            max_atoms_per_token (int, optional): max atoms in a token. Defaults to 20.
            pairformer_dropout (float, optional): dropout ratio for Pairformer. Defaults to 0.0.
            blocks_per_ckpt: number of Pairformer blocks in each activation checkpoint
            distance_bin_start (float, optional): Start of the distance bin range. Defaults to 3.25.
            distance_bin_end (float, optional): End of the distance bin range. Defaults to 52.0.
            distance_bin_step (float, optional): Step size for the distance bins. Defaults to 1.25.
            stop_gradient (bool, optional): Whether to stop gradient propagation. Defaults to True.
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.max_atoms_per_token = max_atoms_per_token
        self.stop_gradient = stop_gradient
        
        self.linear_no_bias_s1 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_s2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )

        self.pairformer_stack = PairformerStack(
            c_z=self.c_z,
            c_s=self.c_s,
            n_blocks=n_blocks,
            dropout=pairformer_dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )

        self.projection = nn.Sequential(
            LayerNorm(self.c_s),
            LinearNoBias(self.c_s, self.c_s //2),
            nn.ReLU(),
            LayerNorm(self.c_s // 2),
            LinearNoBias(self.c_s // 2, 1)
        )
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.input_strunk_ln = LayerNorm(self.c_s)

        # with torch.no_grad():
        #     # Zero init for output layer
        #     for layer in self.projection:
        #         if hasattr(layer, 'weight'):
        #             nn.init.zeros_(layer.weight)

    def forward(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        single_mask: Optional[torch.Tensor] = None,
        use_embedding: bool = True,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "triattention": Optimized tri-attention module
                - "deepspeed": DeepSpeed's fused attention kernel
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": Cuequivariance implementation
            chunk_size (Optional[int], optional): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: Predicted binding scores [..., 1]
        """

        # Optionally treat Protenix trunk outputs as frozen features.
        if self.stop_gradient:
            s_inputs = s_inputs.detach()
            s_trunk = s_trunk.detach()
            z_trunk = z_trunk.detach()

        # Stabilize the single representation before refinement.
        s_trunk = self.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))

        # Allow an ablation where the trunk pair embedding is ignored.
        if not use_embedding:
            if inplace_safe:
                z_trunk *= 0
            else:
                z_trunk = 0 * z_trunk
        print('s_inputs', torch.isnan(s_inputs).any().item())
        print('s_trunk', torch.isnan(s_trunk).any().item())
        print('z_trunk', torch.isnan(s_trunk).any().item())
        print('self.linear_no_bias_s1(s_inputs)', torch.isnan(self.linear_no_bias_s1(s_inputs)).any().item())

        # Build pair initialization from input token features.
        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )

        # Combine fresh pair initialization with Protenix pair features.
        z_trunk = z_init + z_trunk

        if not self.training:
            del z_init
            torch.cuda.empty_cache()

        # Refine single and pair features with head-local Pairformer blocks.
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_trunk,
            pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        
        # Upcast after pairformer
        z_pair = z_pair.to(torch.float32)
        s_single = s_single.to(torch.float32)

        # Pool refined token features into one complex-level score.
        rep = self.pooling(s_single.transpose(-1, -2)).squeeze(-1)
        binding_pred = self.projection(rep)

        return binding_pred

AffinityHead = GatedPoolingHead
BinderHead = PairformerRefinementHead
