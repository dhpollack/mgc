import os
from tabulate import tabulate

OUTPATH = "output/latex"

a = [["Dataset Name", "# of Samples", "Total Length"], ["Balanced", "3,146", "8.74 hours"], ["Eval", "2,574", "7.15 hours"], ["Unbalanced", "224,155", "622.65 hours"], ["Unbalanced - Subset", "19,042", "52.89 hours"]]
if not os.path.exists(OUTPATH):
    try:
        os.makedirs(OUTPATH)
    except:
        raise
with open(os.path.join(OUTPATH, "audioset.tex"), "w") as f:
    f.write(tabulate(a, tablefmt="latex", floatfmt=".2f"))
