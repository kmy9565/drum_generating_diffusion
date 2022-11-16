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

import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn.functional as F
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class SampleDataset(torch.utils.data.Dataset) :
    def __init__(self, metadata_dir, params, df=pd.DataFrame(), size=8000) :
        super().__init__()
        self.metadata_dir = metadata_dir
        self.params = params
        self.df = df
        if (self.df.index == 0).all() :
            self.df = pd.read_csv(self.metadata_dir+"metadata.csv")
        self.df = self.df.replace(np.nan, '', regex=True)
        start = np.random.randint(len(self.df) - size + 1)
        self.df = self.df[start : start + size].reset_index()
        self.length = len(self.df)
        
    def __len__(self) :
        return self.length
    
    def __getitem__(self, idx : int) :
        metadata = self.df.loc[idx, :]
        
        sample_path = metadata["data_dir"]
        categories = metadata["category"].split(",")
        genres = metadata["genre"].split(",")
        styles = metadata["style"].split(",")
        
        signal, _ = torchaudio.load(sample_path, channels_first=True)
        signal = signal[0]
        #signal = F.pad(signal, (10, 0), mode="constant", value=0)
        
        if len(signal) < self.params.audio_len :
            signal = F.pad(signal, (0, self.params.audio_len - len(signal)), mode="constant", value=0)
        end = self.params.audio_len
        signal = signal[:end]
        
        return {
            'audio' : signal,
            #'spectrogram' : None,
            'category' : categories,
            'genre' : genres,
            'style' : styles
        }


class Collator:
    def __init__(self, params):
        self.params = params
    
    def padding(self, list_) :
        pad_length = self.params.tag_padding_len
        if len(list_) < pad_length :
            while len(list_) != pad_length :
                list_.append('')
        else :
            list_ = list_[:pad_length]
        return list_

    def collate(self, minibatch):
        samples_per_frame = self.params.hop_samples
        for sample in minibatch:
            if self.params.unconditional:
                #sample['audio'] = F.pad(sample['audio'], (10, 0), mode="constant", value=0)
                if len(sample['audio']) < self.params.audio_len:
                    sample['audio'] = F.pad(sample['audio'], (0, self.params.audio_len-len(sample['audio'])))

                start = random.randint(0, sample['audio'].shape[-1] - self.params.audio_len)
                end = start + self.params.audio_len
                sample['audio'] = sample['audio'][start:end]
                sample['audio'] = F.pad(sample['audio'], (0, (end - start) - len(sample['audio'])), mode='constant', value=0)
            else:
                #sample['audio'] = F.pad(sample['audio'], (10, 0), mode="constant", value=0)
                #sample['spectrogram'] = F.pad(sample['spectrogram'], (6, 0), mode="constant", value=0)
                if len(sample['spectrogram']) < self.params.crop_mel_frames:
                    sample['audio'] = F.pad(sample['audio'], (0, self.params.audio_len-len(sample['audio'])))
                    sample['spectrogram'] = F.pad(sample['spectrogram'], (0, self.params.crop_mel_frames-len(sample['spectrogram'])))

                start = random.randint(0, sample['spectrogram'].shape[0] - self.params.crop_mel_frames)
                end = start + self.params.crop_mel_frames
                sample['spectrogram'] = sample['spectrogram'][start:end].T
                
                start *= samples_per_frame
                end *= samples_per_frame
                sample['audio'] = sample['audio'][start:end]
                sample['audio'] = np.pad(sample['audio'], (0, (end-start) - len(sample['audio'])), mode='constant')

        audio = torch.stack([sample['audio'] for sample in minibatch if 'audio' in sample])
        category = np.stack([self.padding(sample['category']) for sample in minibatch if 'audio' in sample])
        genre = np.stack([self.padding(sample['genre']) for sample in minibatch if 'audio' in sample])
        style = np.stack([self.padding(sample['style']) for sample in minibatch if 'audio' in sample])
        if self.params.unconditional:
            return {
                'audio': audio,
                #'spectrogram': None,
                'category': category,
                'genre': genre,
                'style': style
            }
        spectrogram = torch.stack([sample['spectrogram'] for sample in minibatch if 'spectrogram' in sample])
        return {
            'audio': audio,
            'spectrogram': spectrogram,
            'category': category,
            'genre': genre,
            'style': style
        }


def from_path(data_dir, params, is_distributed=False):
    """if params.unconditional:
        dataset = UnconditionalDataset(data_dirs)
    else:#with condition
        dataset = ConditionalDataset(data_dirs)"""
    dataset = SampleDataset(data_dir, params)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=int(os.cpu_count()/2),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

