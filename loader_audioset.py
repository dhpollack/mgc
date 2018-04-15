from __future__ import print_function
import torchaudio
import torch.utils.data as data
from torch.autograd import Variable
from torch import stack, load
import os
import errno
import random
import shutil
import json
import csv
import math
import time
from itertools import accumulate, chain

# heavily inspired by:
# https://github.com/patyork/python-voxforge-download/blob/master/python-voxforge-download.ipynb


class AUDIOSET(data.Dataset):
    """`Audioset <http://research.google.com/audioset>`_ Dataset.

    Args:
        TODO: update documentation
        basedir (string): Root directory of dataset.
    """

    NOISES = ("noises", 'http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/Nonspeech.zip')

    SPLITS = ["train", "valid", "test"]

    CLASSES_FILE = "class_labels_targets.csv"

    DATASETS = {
        "balanced": {
            "dir": "processed/balanced",
            "segements_csv": "balanced_train_segments.csv",
        },
        "unbalanced": {
            "dir": "processed/unbalanced",
            "segements_csv": "unbalanced_train_segments.csv",
        }
    }

    AUDIO_EXTS = set(["wav", "mp3", "webm", "m4a", "flac"])
    # set random seed
    random.seed(12345)

    def __init__(self, basedir="data/audioset", transform=None, target_transform=None,
                 dataset="balanced", split="train", use_cache=False, randomize=False,
                 mix_noise=False, mix_prob=.5, mix_vol=.1):

        assert os.path.exists(basedir)

        self.basedir = basedir
        self.dataset = dataset
        self.split = split
        self.use_cache = use_cache
        self.randomize = randomize
        self.mix_noise = mix_noise
        self.mix_prob = mix_prob
        self.mix_vol = lambda: random.uniform(-1, 1) * 0.3 * mix_vol + mix_vol
        self.maxlen = 160013  # precalculated for balanced

        self.transform = transform
        self.target_transform = target_transform

        ds_dict = self.DATASETS[dataset]

        adir = os.path.join(basedir, ds_dict["dir"])
        amanifest = [os.path.join(adir, fn) for fn in os.listdir(adir)]
        num_files = len(amanifest)
        if randomize:
            random.shuffle(amanifest)

        with open(os.path.join(basedir, self.CLASSES_FILE), 'r', newline='') as f_classes:
            csvreader = csv.reader(f_classes, doublequote=True, skipinitialspace=True)
            next(csvreader, None);
            tgt_tags = [[0, "__background__", "no label"]]
            tgt_tags.extend([row for row in csv.reader(f_classes, delimiter=',')])
            tgt_tags = {
                target_key: {
                    "id": target_id,
                    "name": target_name,
                    "label_id": i,
                }
                for i, (target_id, target_key, target_name) in enumerate(tgt_tags)
            }
        self.labels_dict = tgt_tags

        with open(os.path.join(basedir, ds_dict["segements_csv"]), 'r') as f_csv:
            target_keys = set(tgt_tags.keys())
            csvreader = csv.reader(f_csv, doublequote=True, skipinitialspace=True)
            # skip first three rows
            next(csvreader, None);next(csvreader, None);next(csvreader, None);
            segments = [row for row in csvreader]
            # balanced goes from 22160 to 3146
            segments = {
                target_key: {
                    "st": float(st),
                    "fin": float(fin),
                    "tags": tags.split(",")
                }
                for target_key, st, fin, tags in segments
                if set(tags.split(',')).intersection(target_keys)
            }

        labels = []
        for audio_path in amanifest:
            target_key = os.path.basename(audio_path).split(".")[0]
            labels.append([tgt_tags[k]["label_id"] for k in segments[target_key]["tags"]
                           if k in tgt_tags])

        # TODO add cache
        #if self.use_cache:
        #    self.cache = { fn: self._load_data(fn, load_from_cache=False) for fn in amanifest }

        self.data = amanifest
        self.labels = labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (audio, label) where target is index of the target class.
        """

        data_file = self.data[index]

        audio, sr = self._load_data(data_file, self.use_cache)
        target = self.labels[index]
        assert sr == 16000

        if self.split == "train" and self.mix_noise and self.mix_prob > random.random():
            raise NotImplementedError
            #audio = self._add_noise(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self):
        return len(self.data)

    def _load_data(self, data_file, load_from_cache=False):
        ext = data_file.rsplit('.', 1)[1]
        if not load_from_cache:
            if ext in self.AUDIO_EXTS:
                audio, sr = torchaudio.load(data_file, normalization=True)
                if self.transform is not None:
                    audio = self.transform(audio)
                return audio, sr
        else:
            return self.cache[data_file]

    def init_cache(self):
        print("initializing cache...")
        st = time.time()
        self.cache = {}
        for fn in self.data:
            audio, sr = self._load_data(fn, load_from_cache=False)
            self.cache[fn] = (audio, sr)
        print("caching took {0:.2f}s to complete".format(time.time() - st))

    def _add_noise(self, audio):
        raise NotImplementedError
        noise_path = random.choice(self.noises)
        if noise_path in self.cache:
            noise_sig, _ = self.cache[noise_path]
        else:
            noise_sig, _ = torchaudio.load(noise_path, normalization=True)
        diff = audio.size(0) - noise_sig.size(0)
        if diff > 0: # audio longer than noise
            st = random.randrange(0, diff)
            end = audio.size(0) - diff + st
            audio[st:end] += noise_sig * self.mix_vol()
        elif diff < 0:  # noise longer than audio
            st = random.randrange(0, -diff)
            end = st + audio.size(0)
            audio += noise_sig[st:end] * self.mix_vol()
        else:
            audio += noise_sig * self.mix_vol()
        return audio

    def find_max_len(self):
        self.maxlen = 0
        for fp in self.data:
            sig, sr = self._load_data(fp)
            self.maxlen = sig.size(0) if sig.size(0) > self.maxlen else self.maxlen

def bce_collate(batch):
    """Puts batch of inputs into a tensor and labels into a list
       Args:
         batch: (list) [inputs, labels].  In this simple example, I'm just
            assuming the inputs are tensors and labels are strings
       Output:
         minibatch: (Tensor)
         targets: (list[str])
    """

    minibatch, targets = zip(*[(a, b) for (a,b) in batch])
    minibatch = stack(minibatch, dim=0)
    targets = stack(targets, dim=0)
    return minibatch, targets
