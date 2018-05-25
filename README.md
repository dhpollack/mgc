# mgc
Musical Genre Classification - An comparison on different deep learning architectures on musical genre classification of raw audio samples.


## Requirements

pytorch, torchvision, torchaudio, librosa, youtube-dl, ffmpeg


## Instructions

This is easiest with an [Anaconda](https://conda.io) environment. All of the conda programs can be installed with either apt or pip.  I prefer using [miniconda](https://conda.io/miniconda) instead of the full conda install.

0) install requirements, clone git repos, move into base folder
```sh
# install all required library
conda install -c pytorch pytorch torchvision cuda90 ffmpeg
pip install librosa youtube_dl
git clone https://github.com/pytorch/audio.git && cd audio
sudo apt-get install sox libsox-dev libsox-fmt-all
conda install cffi  # OR pip install cffi
python setup.py install
cd ..
git clone https://github.com/dhpollack/mgc.git && cd mgc
```

1) create folder structure for the data, saving models, download label subset csv file

```sh
mkdir -p output/states output/vis data/audioset
wget -P data/audioset https://gist.githubusercontent.com/dhpollack/8a6415c6844178fa85487ffeb3186f11/raw/e0c01e6f6a3a7cdee456042a857d5cfd41ecdd00/class_label_targets.csv
```

2) download the balanced and eval datasets from the audio dataset.

```sh
python -m data.audioset_download --segs-name balanced --download
# note, the path below will be different then below, use the command generated from running the previous line
youtube-dl -ci -f worstaudio -o '/home/david/Programming/repos/dhpollack/mgc/data/audioset/files/balanced/%(id)s.%(ext)s' -a /home/david/Programming/repos/dhpollack/mgc/data/audioset/balanced_urls.txt
python -m data.audioset_download --segs-name balanced
python -m data.audioset_download --segs-name eval --download
youtube-dl -ci -f worstaudio -o '/home/david/Programming/repos/dhpollack/mgc/data/audioset/files/eval/%(id)s.%(ext)s' -a /home/david/Programming/repos/dhpollack/mgc/data/audioset/eval_urls.txt
python -m data.audioset_download --segs-name eval
```

3) for inference-only, download a pretrained model [found here](https://linktomodel) into the output/states folder.  We will use "resnet34_conv_crossentropy_140.pt" for this example

```sh
python -m infer --model-name resnet34_conv --loss-criterion crossentropy --batch-size 10 --load-model output/states/resnet34_conv_crossentropy_140.pt
python -m vis
```

4) to train your own models

```sh
python -m train --model-name resnet34_conv --loss-criterion crossentropy --batch-size 10 --validate --log-interval 50 --save-model
```


## Sample Results

```
insert output of inference here
```

![ResNet34 Losses](output/vis/losses_resnet34_conv_bce_0.json.png)


## Notes

This repo is based on a previous project to do language identification on audio that can be [found here](https://github.com/dhpollack/spokenlanguages)
