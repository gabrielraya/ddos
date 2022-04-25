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
import torch
import torch.nn as nn

# get activation layer
get_act = layers.get_act


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config, sde):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.nf = nf = config.model.nf
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        self.sde = sde
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(layerspp.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale),
                                    nn.Linear(nf, nf))

        channels = [32,64, 128, 256]
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=3, stride=1, bias=False)
        self.dense1 = Dense(input_dim=nf, output_dim=channels[0])
        self.gnorm1 = nn.GroupNorm(num_groups=4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=2, bias=False)
        self.dense2 = Dense(input_dim=nf, output_dim=channels[1])
        self.gnorm2 = nn.GroupNorm(num_groups=32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(nf, channels[2])
        self.gnorm3 = nn.GroupNorm(num_groups=32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=2, bias=False)
        self.dense4 = Dense(nf, channels[3])
        self.gnorm4 = nn.GroupNorm(num_groups=32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(input_dim=nf, output_dim=channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(nf, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(nf, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

    def forward(self, x, t):
        """[(W−K+2P)/S]+1 """

        if self.embedding_type == 'fourier':
            # Obtain the Gaussian random feature embedding for t
            # embd = self.act(self.embed(time_cond))
            embed = self.embed(t)

        # Encoding path
        h1 = self.conv1(x) #(28-3)/1 + 1 = 26
        # print("h1", h1.shape)
        ## Incorporate information from t
        h1+= self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        # The swish activation functio
        h1 = self.act(h1)

        # Do the same for the next layers
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding part
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        _, marginal_prob_std = self.sde.marginal_prob(x, t)
        h = h / marginal_prob_std[:, None, None, None]
        return h
