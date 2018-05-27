import argparse
import json
import csv
from glob import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loader_audioset import AUDIOSET
import mgc_transforms
import torchaudio.transforms as tat

AUDIOSET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "audioset")

DATASET = "balanced"

IMG_SIZE = (10, 5)
CMAP_COLOR = "jet"

T = tat.Compose([
    #tat.PadTrim(self.max_len),
    mgc_transforms.MEL(sr=16000, n_fft=800, hop_length=320, n_mels=224),
    mgc_transforms.BLC2CBL(),
    #mgc_transforms.Scale(),
])
ds = AUDIOSET(AUDIOSET_PATH, transform=T ,dataset=DATASET, num_samples=1)

rev_labeler = {x["label_id"]: x["name"] for _, x in ds.labels_dict.items()}

for sample, label in ds:
    sample.squeeze_()
    sample = sample.numpy()
    sample = np.log(sample)
    sample -= sample.min()

    plt.figure(figsize=IMG_SIZE)
    plt.title("MEL Spectrogram of {} Audio".format(rev_labeler[label[0]].capitalize()))
    plt.imshow(sample, interpolation='nearest',
               aspect='auto', origin="lower", cmap=CMAP_COLOR)
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    logscale = [int(x) for x in np.geomspace(1, 8000, len(labels)).tolist()]
    print(logscale)
    plt.gca().set_yticklabels(logscale)
    plt.xlabel("Time (hops)")
    plt.ylabel("Hz (Log Scale)")
    plt.colorbar()
    plt.savefig("output/vis/spectrogram_{}.png".format(DATASET))
    break
