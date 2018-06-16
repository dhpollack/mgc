import argparse
import json
from glob import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""This file creates visualization of the training and validation losses from
a specified json file or json files.

    Note: default should be run from the base folder as
            "python -m vis.losses"

"""

parser = argparse.ArgumentParser(description='visualize losses')
parser.add_argument('--json-glob', type=str, default="output/losses*.json",
                    help='glob argument for json files')
parser.add_argument('--json-file', type=str, default=None,
                    help='use for a single json file')
parser.add_argument('--save-path', type=str, default="output/vis",
                    help='glob argument for json files')
args = parser.parse_args()

if args.json_file:
    jsonfiles = [args.json_file]
else:
    jsonfiles = glob(args.json_glob)
assert jsonfiles # check that jsonfiles is not empty

C1 = 'blue'
C2 = 'green'
LW = 0.7
ALPHA= 0.2

for jsonfile in jsonfiles:
    print(jsonfile)
    visfile = os.path.join(args.save_path, "{}.png".format(os.path.basename(jsonfile)))
    data = json.load(open(jsonfile, 'r'))
    filename = os.path.basename(jsonfile).split("_")
    nusc = len(filename)
    subtitle = []
    model_name = filename[1].capitalize()
    subtitle.append(model_name)
    loss_type = "BCE" if "bce" in jsonfile else "Cross Entropy"
    subtitle.append(loss_type)
    if "resnet" in model_name.lower():
        preprocess = "Spectrogram" if "conv" in jsonfile else "MFCC"
        subtitle.append(preprocess)
    ds_type = "Unbalanced" if "unbalanced" in jsonfile else "Balanced"
    subtitle.append(ds_type)
    if not "nocache" in jsonfile:
        subtitle.append("w/Cache")
    if not "nonoise" in jsonfile:
        subtitle.append("w/Noise")
    subtitle = ", ".join(subtitle)
    train_losses = np.array(data["train_losses"])
    num_epochs = train_losses.shape[0]
    valid_data = np.array(data["valid_losses"])
    valid_data = valid_data.reshape(num_epochs, -1, 2)
    valid_losses = valid_data[:,:,0]
    valid_acc = valid_data[:,:,1]
    #print(train_losses.shape, valid_losses.shape)
    t = np.arange(train_losses.shape[0])
    train_means = train_losses.mean(axis=1)
    train_stds = train_losses.std(axis=1)
    valid_means = valid_losses.mean(axis=1)
    valid_stds = valid_losses.std(axis=1)
    valid_acc_mean = valid_acc.mean(axis=1) * 100.
    valid_acc_stds = valid_acc.std(axis=1) * 100.
    #print(train_means.shape, valid_means.shape)

    fig, ax1 = plt.subplots()
    ax1.plot(t, train_means, color=C1, linewidth=LW, label="train losses")
    ax1.fill_between(t, train_means-train_stds, train_means+train_stds, facecolor=C1, alpha=ALPHA, interpolate=True)
    ax1.plot(t, valid_means, color=C2, linewidth=LW, label="valid losses")
    ax1.fill_between(t, valid_means-valid_stds, valid_means+valid_stds, facecolor=C2, alpha=ALPHA, interpolate=True)
    plt.title("Training and Validation Losses\n{}".format(subtitle))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Ave Loss (per epoch)")
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # dual axis
    ax2 = ax1.twinx()
    ax2.plot(t, valid_acc_mean, color=C2, linewidth=LW, linestyle='-.', label="valid accuracy")
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0.,100.)
    # legends
    ax1.legend(loc='center right')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(visfile)
    plt.close(fig)
