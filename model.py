# Copyright 2020 LMNT, Inc. All Rights Reserved.
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
# ==============================================================================

import zipfile
import zipimport
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d
Embedding = nn.Embedding


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class EmbeddingTable :
    def __init__(self, vocab, residual_channels) :
        self.vocab = vocab
        self.vocab_category = vocab[0]
        self.vocab_genre = vocab[1]
        self.vocab_style = vocab[2]
        self.embedding_dim = residual_channels

    def category_table(self) :
        num = len(self.vocab_category)
        dim = self.embedding_dim
        return Embedding(num_embeddings=num, embedding_dim=dim)

    def genre_table(self) :
        num = len(self.vocab_genre)
        dim = self.embedding_dim
        return Embedding(num_embeddings=num, embedding_dim=dim)

    def style_table(self) :
        num = len(self.vocab_style)
        dim = self.embedding_dim
        return Embedding(num_embeddings=num, embedding_dim=dim)

    def make_table(self) :
        cat_table = self.category_table().weight
        gen_table = self.genre_table().weight
        sty_table = self.style_table().weight
        return cat_table, gen_table, sty_table


# Tag embedding
class TagEmbedding(nn.Module) :
    def __init__(self, residual_channels, vocab) :
        super().__init__()
        self.residual_channels = residual_channels

        self.vocab_category = vocab[0]
        self.vocab_genre = vocab[1]
        self.vocab_style = vocab[2]

        self.embedding_table = EmbeddingTable(vocab, residual_channels)
        self.cat_table, self.gen_table, self.sty_table = self.embedding_table.make_table()
        self.register_buffer('category embedding', self.cat_table, persistent=False)
        self.register_buffer('genre    embedding', self.gen_table, persistent=False)
        self.register_buffer('style    embedding', self.sty_table, persistent=False)
        
        self.cat_projection1 = Linear(residual_channels, 2* residual_channels)
        self.cat_projection2 = Linear(2*residual_channels, residual_channels)
    
        self.gen_projection1 = Linear(residual_channels, 2*residual_channels)
        self.gen_projection2 = Linear(2*residual_channels, residual_channels)

        self.sty_projection1 = Linear(residual_channels, 2*residual_channels)
        self.sty_projection2 = Linear(2*residual_channels, residual_channels)

        self.encoding_mu1  = Linear(3*residual_channels, residual_channels)
        self.encoding_mu2  = Linear(residual_channels, residual_channels)

        self.encoding_var1 = Linear(3*residual_channels, residual_channels)
        self.encoding_var2 = Linear(residual_channels, residual_channels)

    def category_embedding(self, list_) :
        embedded = []
        for vector in list_ :
            vectors = []
            for category in vector :
                try :
                    idx = self.vocab_category[category]
                except :
                    idx = self.vocab_category[None]
                vectors.append(self.cat_table[idx])
            embedded.append(torch.mean(torch.stack(vectors, dim=0), 0))
        return torch.stack(embedded, 0) 

    def genre_embedding(self, list_) :
        embedded = []
        for vector in list_ :
            vectors = []
            for genre in vector :
                try :
                    idx = self.vocab_genre[genre]
                except :
                    idx = self.vocab_genre[None]
                vectors.append(self.gen_table[idx])
            embedded.append(torch.mean(torch.stack(vectors, dim=0), 0))
        return torch.stack(embedded, 0) 

    def style_embedding(self, list_) :
        embedded = []
        for vector in list_ :
            vectors = []
            for style in vector :
                try :
                    idx = self.vocab_style[style]
                except :
                    idx = self.vocab_style[None]                
                vectors.append(self.sty_table[idx])
            embedded.append(torch.mean(torch.stack(vectors, dim=0), 0))
        return torch.stack(embedded, 0) 

    def encode(self, embedded) :
        mu  = self.encoding_mu2(F.relu(self.encoding_mu1(embedded)))
        var = self.encoding_var2(F.relu(self.encoding_var1(embedded)))
        return mu, var

    def reparameterize(self, mu, var) :
        std = torch.exp(0.5 * var)
        eps = torch.rand_like(std)
        return mu + std * eps
    
    def forward(self, data) :
        category = data["category"]
        genre    = data["genre"]
        style    = data["style"]

        embedded_cat = self.category_embedding(category)
        embedded_gen = self.genre_embedding(genre)
        embedded_sty = self.style_embedding(style)

        embedded_cat = silu(self.cat_projection1(embedded_cat))
        embedded_cat = silu(self.cat_projection2(embedded_cat))
        
        embedded_gen = silu(self.gen_projection1(embedded_gen))
        embedded_gen = silu(self.gen_projection2(embedded_gen))

        embedded_sty = silu(self.sty_projection1(embedded_sty))
        embedded_sty = silu(self.sty_projection2(embedded_sty))

        embedded = torch.cat([embedded_cat, embedded_gen, embedded_sty], dim=1)
        mu, var = self.encode(embedded)
        z = self.reparameterize(mu, var)
        return z


# Diffusion step embedding
class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()  
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        self.tag_projection = Conv1d(residual_channels, residual_channels, 1)

        if not uncond: # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, tag, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        tag = tag.unsqueeze(-1)
        y = x + diffusion_step + tag
        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params, vocab):
        super().__init__()
        self.params = params
        self.vocab = vocab
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.tag_embedding = TagEmbedding(params.residual_channels, self.vocab)

        if self.params.unconditional: # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

        self.residual_layers = nn.ModuleList([
             ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
             for i in range(params.residual_layers)
            ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, tags, diffusion_step, spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler: # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)

        tag_embedded = self.tag_embedding(tags)
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, tag_embedded, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


if __name__=="__main__" :
    from params import params
    model = DiffWave(params, )
    count = 0
    for name, param in model.named_parameters():
        count+=1
        if count==1:
            print(param.size())
            print(name)    
    print(count)