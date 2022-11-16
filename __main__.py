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

import torchaudio

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train, train_distributed
from params import params


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def main(args, vocab, naive):
    replica_count = device_count()
    if replica_count > 1:
        if params.batch_size % replica_count != 0:
            raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        params.batch_size = params.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
    else:
        train(args, params, vocab, naive)


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

from torchsummary import summary as summary
from dataset import from_path

if __name__ == '__main__' :
    model_dir = ".\\model_checkpoint"
    metadata_dir = "D:\\Study\\인턴\\Intern project\\"
    tag_dir = "./tags.txt"

    vocab = make_vocab(tag_dir)

    args = dict(model_dir=model_dir, metadata_dir=metadata_dir, max_steps=None, fp16=False)
    naive = False # naive learning : training from vanilla version
    main(args, vocab, naive)


## Argument requires ... [Model_dir, Data_dir, (Optionally) --max_steps, --fp16]        