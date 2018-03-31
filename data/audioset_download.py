import argparse
import os
import csv
from urllib.request import urlretrieve

"""
    This script downloads the Google Audioset dataset.  Optionally with a target
    list of labels instead of the entire list
"""

parser = argparse.ArgumentParser(description='Google Audioset Download Script')
parser.add_argument('--segs-name', type=str, default='balanced',
                    help='balanced / unbalanced / eval')
parser.add_argument('--target-list', type=str, default='class_labels_targets.csv',
                    help='file name of target list of labels (relative to BASEDIR)')
parser.add_argument('--download', action='store_true',
                    help='get youtube-dl command to download files.')
args = parser.parse_args()

YT_PREFIX = 'https://www.youtube.com/watch?v='

BASEDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'audioset')

CLASS_LABELS_CSV = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
EVAL_SEGS_CSV = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv'
BAL_TRAIN_SEGS_CSV = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv'
UNBAL_TRAIN_SEGS_CSV = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'

CSV_FILES = [CLASS_LABELS_CSV, EVAL_SEGS_CSV, BAL_TRAIN_SEGS_CSV, UNBAL_TRAIN_SEGS_CSV]

try:
    os.makedirs(BASEDIR)
except FileExistsError:
    print("{} already exists".format(BASEDIR))

for uri_csv in CSV_FILES:
    fn_csv = os.path.join(BASEDIR, os.path.basename(uri_csv))
    if not os.path.exists(fn_csv):
        print('Downloading {}'.format(uri_csv))
        urlretrieve(uri_csv, fn_csv)

tgt_tags_fn = args.target_list if args.target_list else os.path.basename(CLASS_LABELS_CSV)
tgt_tags_fn = os.path.join(BASEDIR, tgt_tags_fn)

with open(tgt_tags_fn, 'r', newline='') as f:
    # list in format [['244', '/m/0m0jc', 'Electronica'], ...]
    tgt_tags = [row for row in csv.reader(f, delimiter=',')]
    column_headers = tgt_tags.pop(0) # saving this but we probably won't use it

tags_idx, tags_code, tags_name = zip(*tgt_tags)
tags_code_set = set(tags_code)

if args.segs_name.lower() == 'unbalanced':
    segs_name = args.segs_name.lower()
    segs_csv = UNBAL_TRAIN_SEGS_CSV
elif args.segs_name.lower() == 'eval':
    segs_name = args.segs_name.lower()
    segs_csv = EVAL_SEGS_CSV
else:  # default val: balanced
    segs_name = 'balanced'
    segs_csv = BAL_TRAIN_SEGS_CSV

manifest_fn = os.path.basename(segs_csv)
manifest_fn = os.path.join(BASEDIR, manifest_fn)

with open(manifest_fn, 'r', newline='') as f:
    csvreader = csv.reader(f, doublequote=True, skipinitialspace=True)
    # skip first three rows
    next(csvreader, None);next(csvreader, None);next(csvreader, None);
    segments = [row for row in csvreader]
    # balanced goes from 22160 to 3146
    segments = [(idx, st, fin, tags.split(',')) for idx, st, fin, tags in segments
                if set(tags.split(',')).intersection(tags_code_set)]
    print("{} files were found".format(len(segments)))

yt_idxes, _, _, _ = zip(*segments)

yt_urls = ['{}{}'.format(YT_PREFIX, idx) for idx in yt_idxes]

yt_urls_file = os.path.join(BASEDIR, '{}_urls.txt'.format(segs_name))
with open(yt_urls_file, 'w') as f:
    f.write("\n".join(yt_urls))

files_dir = os.path.join(BASEDIR, 'files', segs_name)
try:
    os.makedirs(files_dir)
except FileExistsError:
    print("{} already exists".format(files_dir))

raw_yt_files = os.listdir(files_dir)

if args.download:
    yt_dl = "youtube-dl -ci -f worstaudio \\ \n-o '{}/%(id)s.%(ext)s' \\ \n-a {}".format(files_dir, yt_urls_file)
    # youtube-dl -ci -f worstaudio -o '/home/david/Programming/repos/dhpollack/mgc/data/audioset/files/%(id)s.%(ext)s' -a /home/david/Programming/repos/dhpollack/mgc/data/audioset/urls.txt
    print("run the following:\n\n{}".format(yt_dl))
else:
    processed_dir = os.path.join(BASEDIR, 'processed', segs_name)
    try:
        os.makedirs(processed_dir)
    except FileExistsError:
        print("{} already exists".format(processed_dir))
    for raw_yt_fn in raw_yt_files:
        raw_yt_fn, raw_yt_ext = raw_yt_fn.split('.')
        yt_idx = yt_idxes.index(raw_yt_fn)
        _, st, fn, tags = segments[yt_idx]
        st, fn = float(st), float(fn)
        dur = fn - st
        infile = os.path.join(files_dir, raw_yt_fn)
        outfile = os.path.join(processed_dir, raw_yt_fn + '.wav')
        cmd = "ffmpeg -ss {} -t {} -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(st, dur, infile, outfile)
        print(cmd)
        os.system(cmd)
        break
# ffmpeg -ss 30 -t 10 -i files/zyXa2tdBTGc.webm -acodec pcm_s16le -ac 1 -ar 16000 processed/zyXa2tdBTGc.wav
