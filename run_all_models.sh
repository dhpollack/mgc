#! /bin/bash

python -m train --model-name squeezenet --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet34_conv --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet34_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet101_conv --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet101_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name attn --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
#python -m train --model-name resnet34_conv --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name resnet34_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name resnet101_conv --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name resnet101_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name attn --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
python -m train --model-name resnet34_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet34_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet101_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet101_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name attn --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet34_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
python -m train --model-name resnet34_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
python -m train --model-name resnet101_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name resnet101_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
python -m train --model-name attn --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --use-cache
python -m train --model-name resnet34_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet34_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999
python -m train --model-name resnet101_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999
#python -m train --model-name resnet101_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999
python -m train --model-name attn --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999
#python -m train --model-name resnet34_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name resnet34_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name resnet101_conv --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name resnet101_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999 --use-cache
#python -m train --model-name attn --batch-size 100 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 48000 --chkpt-interval 99999 --use-cache
python -m train --model-name resnet34_conv --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --noises-dir /home/ubuntu/spokenlanguages/data/voxforge/processed/noises
python -m train --model-name resnet34_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --noises-dir /home/ubuntu/spokenlanguages/data/voxforge/processed/noises
python -m train --model-name resnet101_conv --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --noises-dir /home/ubuntu/spokenlanguages/data/voxforge/processed/noises
python -m train --model-name resnet101_mfcc_librosa --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --noises-dir /home/ubuntu/spokenlanguages/data/voxforge/processed/noises
python -m train --model-name attn --batch-size 100 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 48000 --chkpt-interval 99999 --noises-dir /home/ubuntu/spokenlanguages/data/voxforge/processed/noises
#python -m train --model-name bytenet --batch-size 10 --validate --log-interval 20 --dataset balanced --loss-criterion bce --max-len 160000 --chkpt-interval 99999
python -m train --model-name bytenet --batch-size 10 --validate --log-interval 20 --dataset unbalanced --loss-criterion bce --max-len 160000 --chkpt-interval 99999
python -m train --model-name bytenet --batch-size 10 --validate --log-interval 20 --dataset unbalanced --loss-criterion crossentropy --max-len 160000 --chkpt-interval 99999
