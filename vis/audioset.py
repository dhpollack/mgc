import argparse
import json
import csv
from glob import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""This file creates visualization of the Audioset dataset.

    Note: default should be run from the base folder as
            "python -m vis.audioset"

"""

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
parser.add_argument('--dataset', type=str, default="balanced",
                    help='which Audioset dataset to use balanced / eval / unbalanced')
parser.add_argument('--add-no-label', action='store_true',
                    help='add a label for "no label" or background noise')
args = parser.parse_args()

VIS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "output", "vis")

AUDIOSET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "audioset")

TARGETS_PATH = os.path.join(AUDIOSET_PATH, "class_labels_targets.csv")

ALL_LABELS_PATH = os.path.join(AUDIOSET_PATH, "class_labels_indices.csv")

DATASETS = {
    "balanced": {"csvprefix": "train_segments"},
    "eval": {"csvprefix": "segments"},
    "unbalanced_subset": {"csvprefix": "train_segments_cropped"},
    "unbalanced": {"csvprefix": "train_segments"},
}

data = {}
data["metadata"] = {}

# get total number of labels
with open(ALL_LABELS_PATH, 'r', newline='') as f_classes:
    csvreader = csv.reader(f_classes, doublequote=True, skipinitialspace=True)
    next(csvreader, None);
    total_labels = len([row for row in csvreader])

# get available classes for the Audioset dataset and add a 'no label' class
with open(TARGETS_PATH, 'r', newline='') as f_classes:
    csvreader = csv.reader(f_classes, doublequote=True, skipinitialspace=True)
    next(csvreader, None);
    if args.add_no_label:
        tgts_dict = [[0, "__background__", "no label"]]
    else:
        tgts_dict = []
    tgts_dict.extend([row for row in csvreader])
    tgts_dict = {
        target_key: {
            "id": target_id,
            "name": target_name,
            "label_id": i,
        }
        for i, (target_id, target_key, target_name) in enumerate(tgts_dict)
    }
    target_keys = set(tgts_dict.keys())
#print(tgts_dict)
data["metadata"].update({"labels": (total_labels, len(tgts_dict))})
print("Using {} of {} labels in Audioset".format(len(tgts_dict), total_labels))

for ds in DATASETS.keys():
    if "_" in ds:
        ds_prefix = ds.split("_")[0]
    else:
        ds_prefix = ds
    targets_counter = {}
    segments_path = os.path.join(AUDIOSET_PATH, "{}_{}.csv".format(ds_prefix, DATASETS[ds]["csvprefix"]))
    # get the info on each segment (audio clip), including the label
    with open(segments_path, 'r') as f_csv:
        csvreader = csv.reader(f_csv, doublequote=True, skipinitialspace=True)
        # skip first three rows
        next(csvreader, None);next(csvreader, None);next(csvreader, None);
        segments = [row for row in csvreader]
        total_segments = len(segments)
        segments = {
            yt_id: {
                "st": float(st),
                "fin": float(fin),
                "tgts": [x for x in tgts.split(",") if x in target_keys],
            }
            for yt_id, st, fin, tgts in segments
            if set(tgts.split(',')).intersection(target_keys)
        }
        target_segments = len(segments)
        print("Using {} of {} samples in {}".format(target_segments, total_segments, ds))
        for k, v in segments.items():
            segments[k]["tgts_name"] = []
            segments[k]["tgts_id"] = []
            for tgt in v["tgts"]:
                tgt_name = tgts_dict[tgt]["name"]
                tgt_id = tgts_dict[tgt]["label_id"]
                segments[k]["tgts_name"].append(tgt_name)
                segments[k]["tgts_id"].append(tgt_id)
                if tgt_id not in targets_counter:
                    targets_counter[tgt_id] = 1
                else:
                    targets_counter[tgt_id] += 1
        data[ds] = segments
        data["metadata"].update({ds: (total_segments, target_segments)})
        index, counts = zip(*sorted([(k, v) for k, v in targets_counter.items()]))
        # plot stuff
        plt.figure()
        plt.title("Audioset Label Count for {} Dataset".format(ds_prefix.capitalize()))
        plt.xlabel("Label IDs")
        plt.ylabel("Count")
        plt.bar(index, counts)
        plt.savefig(os.path.join(VIS_PATH, "counts_{}.png".format(ds)))

with open(os.path.join(VIS_PATH, "audioset_metadata.json"), "w") as f:
    json.dump(data["metadata"],f)
