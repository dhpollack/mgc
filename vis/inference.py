import argparse
import json
from glob import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import pdb


"""This file creates a latex table of the inference results.

    Note: default should be run from the base folder as
            "python -m vis.inference --output-latex"

"""

parser = argparse.ArgumentParser(description='visualize losses')
parser.add_argument('--json-glob', type=str, default="output/inference*.json",
                    help='glob argument for json files')
parser.add_argument('--json-file', type=str, default=None,
                    help='use for a single json file')
parser.add_argument('--save-path', type=str, default="output/vis",
                    help='glob argument for json files')
parser.add_argument("--output-path", type=str, default="output/latex")
parser.add_argument('--output-latex', action="store_true")
args = parser.parse_args()

if args.json_file:
    jsonfiles = [args.json_file]
else:
    jsonfiles = glob(args.json_glob)
assert jsonfiles # check that jsonfiles is not empty

all_files = [('Network Type', 'Accuracy')]
for jf in jsonfiles:
    data = json.load(open(jf, "r"))
    data = np.array(data)
    nlabels = data[:, 0].sum()
    tp = data[:, 2].sum()
    acc = 100. * tp / nlabels
    fn = os.path.basename(jf)
    fn_split = fn.split("_")
    i = 0
    nettypename = []
    nettypename.append(fn_split[i + 1].capitalize())
    if fn_split[i + 2] == "conv":
        nettypename.append("Spectrogram")
    elif fn_split[i + 2] == "mfcc":
        nettypename.append("MFCC")
        i += 1  # skip "librosa"
    else:
        specname = ""
        i -= 1  # next i will be third item in list
    nettypename.append("BCE" if fn_split[i + 3] == "bce" else "Cross Entropy")
    print(fn_split[i + 4], fn_split)
    nettypename.append(fn_split[i + 4].capitalize())
    nettypename.append("No Noise" if fn_split[i + 5] == "nonoise" else "Noise Added")
    nettypename.append("No Cache" if fn_split[i + 6] == "nocache" else "w/Cache")
    all_files.append((", ".join(nettypename), acc))

print(all_files)

if args.output_latex:
    from tabulate import tabulate
    s = tabulate(all_files, tablefmt="latex", floatfmt=".2f")
    if not os.path.exists(args.output_path):
        try:
            os.makedirs(args.output_path)
        except:
            raise
    with open(os.path.join(args.output_path, "inference_table.tex"), "w") as f:
        f.write(s)

#pdb.set_trace()
