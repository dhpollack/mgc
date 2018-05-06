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
from glob import glob

# heavily inspired by:
# https://github.com/patyork/python-voxforge-download/blob/master/python-voxforge-download.ipynb


class AUDIOSET(data.Dataset):
    """`Audioset <http://research.google.com/audioset>`_ Dataset.

    Args:
        TODO: update documentation
        basedir (string): Root directory of dataset.
    """

    #NUM_VALID_SAMPLES = 100

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
        },
        "eval": {
            "dir": "processed/eval",
            "segements_csv": "eval_segments.csv",
        }
    }

    AUDIO_EXTS = set(["wav", "mp3", "webm", "m4a", "flac"])
    # set random seed
    random.seed(12345)

    def __init__(self, basedir="data/audioset", transform=None, target_transform=None,
                 dataset="balanced", split="train", use_cache=False, randomize=False,
                 noises_dir=None, mix_prob=.5, mix_vol=.2,
                 otype='long', num_samples=None):

        assert os.path.exists(basedir)

        self.basedir = basedir
        self.dataset = dataset
        self.split = split
        self.use_cache = use_cache
        self.randomize = randomize
        self.noises_dir = noises_dir
        self.mix_prob = mix_prob
        self.mix_vol = lambda: random.uniform(-1, 1) * 0.3 * mix_vol + mix_vol
        self.maxlen = 160013  # precalculated for balanced
        self.data = {}
        self.labels = {}
        self.cache = {}

        self.transform = transform
        self.target_transform = target_transform

        # get available classes for the Audioset dataset and add a 'no label' class
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
        # setup noise files
        if self.noises_dir:
            self.noises = [fn for fn in glob(os.path.join(self.noises_dir, '*.*'))]

        # get list of files from appropriate dataset (balanced/unbalanced)
        ds_dict = self.DATASETS[dataset]
        amanifest, labels = self._init_set(ds_dict, self.randomize, num_samples)
        self.data[self.split] = amanifest
        self.labels[self.split] = labels
        # only initialize cache if first file not found in cache
        if self.use_cache and self.data[self.split][0] not in self.cache:
            self.init_cache()

    def init_cache(self):
        print("initializing cache...")
        st = time.time()
        for fn in self.data[self.split]:
            audio, sr = self._load_data(fn, load_from_cache=False)
            self.cache[fn] = (audio, sr)
        print("caching took {0:.2f}s to complete".format(time.time() - st))

    def find_max_len(self):
        """
            This function finds the maximum length of all audio samples in the dataset.
            This is done naively and not necessarily efficiently.  Perhaps using something
            like soxi would be a better solution.
        """
        self.maxlen = 0
        for fp in self.data[self.split]:
            sig, sr = self._load_data(fp)
            self.maxlen = sig.size(0) if sig.size(0) > self.maxlen else self.maxlen

    def set_split(self, split, num_samples=None):
        map_split = {
            "train":"balanced",
            "valid":"eval",
            "test":"unbalanced",
        }
        if split not in self.labels:  # and not in self.data
            # do special stuff for validation
            randomize = True if split == "valid" else self.randomize
            limit = num_samples if num_samples else None
            #limit = self.NUM_VALID_SAMPLES if split == "valid" else None
            # begin initialization
            ds_name = map_split[split]
            ds_dict = self.DATASETS[ds_name]
            d, l = self._init_set(ds_dict, randomize, limit)
            self.data[split] = d
            self.labels[split] = l
        self.split = split
        # only initialize cache if first file not found in cache
        if self.use_cache and self.data[self.split][0] not in self.cache:
            self.init_cache()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (audio, label) where target is index of the target class.
        """

        # get targets
        target = self.labels[self.split][index]

        # get the filename at a particular index
        fn = self.data[self.split][index]

        # load either the file or the cache output of the file after any transformations
        # Note: audio transformations are done in the _load_data function
        audio, sr = self._load_data(fn, self.use_cache)
        assert sr == 16000  # this can be removed for a generic algo

        # transform labels
        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self):
        return len(self.data[self.split])

    def _load_data(self, data_file, load_from_cache=False):
        """
            This function takes a filename as input and returns the audio data
            after transformations and the sample rate of the audio.  We will also
            lower the volume of any samples that are too loud.

        Args:
            data_file (str):  path to audio file.  For caching, we use the file
                path as the key in a cache dictionary.

            load_from_cache (bool):  switch to load from the cache or not.  In the
                caching function, this is set to false, so the cache can be created.

        Returns:
            tuple (audio, sr):  The audio will be the tensor that goes into the
                network as an input.  The sample rate, sr, is not really used,
                but I check that the sample rates are all the same so if we add
                any recorded noises, the tensors will align correctly.

        """
        ext = data_file.rsplit('.', 1)[1]
        if not load_from_cache:
            if ext in self.AUDIO_EXTS:
                audio, sr = torchaudio.load(data_file, normalization=True)
                # check for max volume, this is relatively slow
                amax = audio.max()
                if amax > .7:
                    audio *= .7 / amax
                if self.split == "train" and self.noises_dir and self.mix_prob > random.random():
                    audio = self._add_noise(audio, sr)
                if self.transform is not None:
                    audio = self.transform(audio)
                return audio, sr
        else:
            return self.cache[data_file]

    def _add_noise(self, audio, audio_sr):
        """
            This is a simple additive noise mixer.  The signal of a randomly selected
            noise is multiplied by a volume and then added to the input audio signal.

        """
        start = time.time()
        noise_path = random.choice(self.noises)
        noise_sig, noise_sr = torchaudio.load(noise_path, normalization=True)
        assert noise_sr == audio_sr
        diff = audio.size(0) - noise_sig.size(0)
        if diff > 0: # audio longer than noise
            st = random.randrange(0, diff)
            end = audio.size(0) - diff + st
            audio[st:end] += noise_sig * self.mix_vol()
        elif diff < 0:  # noise longer than audio
            st = random.randrange(0, -diff)  # diff is a negative number
            end = st + audio.size(0)
            audio += noise_sig[st:end] * self.mix_vol()
        else:
            audio += noise_sig * self.mix_vol()
        #assert audio.min() > -1.  # this significantly slows loading
        finish = time.time()
        print("audio mixing took {0:.05f}".format(finish - start))
        return audio

    def _init_set(self, ds_dict, randomize, limit=None):
        adir = os.path.join(self.basedir, ds_dict["dir"])
        amanifest = [fn for fn in glob(os.path.join(adir, '*.*'))]
        num_files = len(amanifest)
        if randomize:
            random.shuffle(amanifest)
        if limit:
            amanifest = amanifest[:limit]
        # get the info on each segment (audio clip), including the label
        with open(os.path.join(self.basedir, ds_dict["segements_csv"]), 'r') as f_csv:
            target_keys = set(self.labels_dict.keys())
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
        # for each audio clip create a list of the labels as integers (i.e. [3, 45])
        labels = []
        for audio_path in amanifest:
            target_key = os.path.basename(audio_path).split(".")[0]
            labels.append([self.labels_dict[k]["label_id"] for k in segments[target_key]["tags"]
                           if k in self.labels_dict])
        return amanifest, labels

def bce_collate(batch, type='long'):
    """Puts batch of inputs into a tensor and labels into a list
       Args:
         batch: (list) [inputs, labels].  In this simple example, I'm just
            assuming the inputs are tensors and labels are strings
       Output:
         minibatch: (Tensor) N x *, where N is the size of the minibatch
         targets: (Tensor) N x Classes, where Classes is the total number of classes
    """

    minibatch, targets = zip(*[(a, b) for (a,b) in batch])
    minibatch = stack(minibatch, dim=0)
    targets = stack(targets, dim=0)
    return minibatch, targets
