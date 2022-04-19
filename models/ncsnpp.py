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

from . import utils, layers, layerspp
import torch.nn as nn

# get activation layer
get_act = layers.get_act


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.nf = nf = config.model.nf
        self.embedding_type = embedding_type = config.model.embedding_type.lower()

        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(layerspp.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale),
                                    nn.Linear(nf, nf))

    def forward(self, time_cond):

        if self.embedding_type == 'fourier':
            # Obtain the Gaussian random feature embedding for t
            # used_sigmas = time_cond
            t_embd = self.act(self.embed(time_cond))
        return t_embd