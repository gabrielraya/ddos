# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""

from . import layers

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels
    paper: https://arxiv.org/abs/2006.10739
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embedding_size//2)*scale, requires_grad=False)

    def forward(self, t):
        """
        Represents an encoding for time step  t so that the score network can condition this encoding
        :param t: time step vector
        :return: a basic input mapping f(v) = [sin2piv, cos2piv,]^T
        """
        v = t[:, None] * self.W[None, :]
        x_proj =  2 * np.pi * v
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)