# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

from models.protenix import Protenix
from models.scoring_heads import resolve_scoring_head_class
from utils.torch_utils import autocasting_disable_decorator


class ComplexScoringModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.N_cycle = self.configs.model.N_cycle
        self.protenix_model = Protenix(configs)
        head_name = getattr(self.configs.model, "scoring_head_name", "gated_pooling")
        head_class = resolve_scoring_head_class(head_name)
        self.scoring_head_name = head_name
        self.scoring_head = head_class(**configs.model.binder_head)
        # Backward-compatible attribute name for older checkpoints/code paths.
        self.Binder_head = self.scoring_head
        
    def forward(
        self,
        input_feature_dict: dict[str, Any]
    ):
        # Use chunked inference only when activations do not need gradients.
        inplace_safe = not torch.is_grad_enabled()
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        # Run the Protenix trunk up to the pairformer outputs.
        s_inputs, s, z = self.protenix_model.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size
        )
    
        # Build the residue-pair mask expected by the scoring head.
        pair_mask = input_feature_dict['seq_mask'].unsqueeze(2) * input_feature_dict['seq_mask'].unsqueeze(1)

        # Score the complex from trunk single and pair representations.
        output = self.run_head(
            s_inputs,
            s,
            z,
            single_mask=input_feature_dict['seq_mask'],
            pair_mask=pair_mask,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        
        return output


    def run_head(self, *args, **kwargs):
        # Disable AMP around the scoring head when the config requests full precision there.
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.scoring_head
        )(*args, **kwargs)


DistilledBinder = ComplexScoringModel
