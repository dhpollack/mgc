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
parser.add_argument('--json-glob', type=str, default="output/*.json",
                    help='glob argument for json files')
parser.add_argument('--save-path', type=str, default="output/vis",
                    help='glob argument for json files')
args = parser.parse_args()

jsonfiles = glob(args.json_glob)

assert jsonfiles # check that jsonfiles is not empty

for jsonfile in jsonfiles:
    print(jsonfile)
    visfile = os.path.join(args.save_path, "{}.png".format(os.path.basename(jsonfile)))
    data = json.load(open(jsonfile, 'r'))
    filename = os.path.basename(jsonfile).split("_")
    nusc = len(filename)
    model_name = filename[1].capitalize()
    loss_type = filename[-2].upper() if len(filename[-2]) < 4 else filename[-2].capitalize() 
    subtitle = "{} trained with {} loss".format(model_name, loss_type)
    if nusc == 5:
        preprocess = "Spectrogram" if filename[2] == "conv" else "MFCC"
        subtitle += " ({})".format(preprocess)

    train_losses = np.array(data["train_losses"])
    valid_losses = np.array(data["valid_losses"])
    #print(train_losses.shape, valid_losses.shape)
    t = np.arange(train_losses.shape[0])
    train_means = train_losses.mean(axis=1)
    valid_means = valid_losses.mean(axis=1)
    #print(train_means.shape, valid_means.shape)

    plt.figure()
    plt.plot(t, train_means, color='blue', label="train losses")
    plt.plot(t, valid_means, color='green', label="valid losses")
    plt.title("Training and Validation Losses\n{}".format(subtitle))
    plt.xlabel("Epoch")
    plt.ylabel("Ave Loss (per epoch)")
    plt.legend()
    plt.savefig(visfile)
