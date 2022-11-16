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
import os
import torch
import torchaudio

from argparse import ArgumentParser

from params import AttrDict, params as base_params
from model import DiffWave


models = {}

def predict(spectrogram=None, vocab=None, tags=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
    # Lazy load model.
    if not model_dir in models:
        if os.path.exists(f'{model_dir}/weights.pt'):
            checkpoint = torch.load(f'{model_dir}/weights.pt')
        else:
            checkpoint = torch.load(model_dir)
        model = DiffWave(AttrDict(base_params), vocab).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        models[model_dir] = model

    model = models[model_dir]
    model.params.override(params)

    with torch.no_grad():
        # Change in notation from the DiffWave paper for fast sampling.
        # DiffWave paper -> Implementation below
        # --------------------------------------
        # alpha -> talpha
        # beta -> training_noise_schedule
        # gamma -> alpha
        # eta -> beta
        training_noise_schedule = np.array(model.params.noise_schedule)
        inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)


        if not model.params.unconditional:
            if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(device)
            audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
        else:
            audio = torch.randn(1, params.audio_len, device=device)
        noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            audio = c1 * (audio - c2 * model(audio, tags, torch.tensor([T[n]], device=audio.device), spectrogram).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
    return audio, model.params.sample_rate


def main(args, vocab):
    if args["spectrogram_path"] :
        spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
    else:
        spectrogram = None
    audio, sr = predict(spectrogram, vocab=vocab, tags=args["tags"], model_dir=args["model_dir"], 
                        fast_sampling=args["fast"], params=base_params)
    torchaudio.save(args["output"], audio.cpu(), sample_rate=sr)


def make_vocab(tag_dir) :
    f = open(tag_dir, 'r')
    vocab_category = {}; vocab_genre = {}; vocab_style = {}
    
    lines = f.readlines()
    for i in range(len(lines)) :
        list_ = lines[i].split(',')
        if i==0 :   vocab_category = {cat: i+1 for i, cat in enumerate(list_)}; vocab_category[None] = 0
        elif i==1 : vocab_genre    = {gen: i+1 for i, gen in enumerate(list_)}; vocab_genre[None] = 0
        else :      vocab_style    = {sty: i+1 for i, sty in enumerate(list_)}; vocab_style[None] = 0
    
    vocab = vocab_category, vocab_genre, vocab_style
    return vocab


if __name__ == '__main__':
    model_dir = ".\\model_checkpoint"
    metadata_dir = "D:\\Study\\인턴\\Intern project\\"
    tag_dir = "./tags.txt"
    output = "inference_test_kick_punch_hardcore_1.wav"

    tags = dict(category=["snare"], genre=["dubstep"], style=["punch"])
    vocab = make_vocab(tag_dir)

    args = dict(model_dir=model_dir, spectrogram_path=False, tags=tags, output=output, fast=False)
    """
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('model_dir',
        help='directory containing a trained model (or full path to weights.pt file)')
    parser.add_argument('--spectrogram_path', '-s',
        help='path to a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('--output', '-o', default='output.wav',
        help='output file name')
    parser.add_argument('--fast', '-f', action='store_true',
        help='fast sampling procedure')
    """
    main(args, vocab)
