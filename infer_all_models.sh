#! /bin/bash

# resnet34 networks
python -m infer --model-name resnet34_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet34_conv_bce_balanced_final.pt
python -m infer --model-name resnet34_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet34_conv_bce_balanced_final_noisy.pt
python -m infer --model-name resnet34_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet34_conv_bce_unbalanced_final.pt
python -m infer --model-name resnet34_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet34_conv_bce_unbalanced_final_cached.pt
python -m infer --model-name resnet34_conv --loss-criterion crossentropy --batch-size 50 --max-len 48000 --load-model output/states/resnet34_conv_crossentropy_balanced_final_noisy.pt
python -m infer --model-name resnet34_conv --loss-criterion crossentropy --batch-size 50 --max-len 48000 --load-model output/states/resnet34_conv_crossentropy_unbalanced_final.pt
python -m infer --model-name resnet34_mfcc_librosa --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet34_mfcc_librosa_bce_balanced_final.pt
python -m infer --model-name resnet34_mfcc_librosa --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet34_mfcc_librosa_bce_unbalanced_final.pt
python -m infer --model-name resnet34_mfcc_librosa --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet34_mfcc_librosa_bce_unbalanced_final_cached.pt
# resnet101 networks
python -m infer --model-name resnet101_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet101_conv_bce_balanced_final.pt
python -m infer --model-name resnet101_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet101_conv_bce_balanced_final_noisy.pt
python -m infer --model-name resnet101_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet101_conv_bce_unbalanced_final.pt
python -m infer --model-name resnet101_conv --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet101_conv_bce_unbalanced_final_cached.pt
python -m infer --model-name resnet101_conv --loss-criterion crossentropy --batch-size 50 --max-len 48000 --load-model output/states/resnet101_conv_crossentropy_balanced_final_noisy.pt
python -m infer --model-name resnet101_conv --loss-criterion crossentropy --batch-size 50 --max-len 48000 --load-model output/states/resnet101_conv_crossentropy_unbalanced_final.pt
python -m infer --model-name resnet101_mfcc_librosa --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet101_mfcc_librosa_bce_balanced_final.pt
python -m infer --model-name resnet101_mfcc_librosa --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/resnet101_mfcc_librosa_bce_unbalanced_final.pt
# attn networks
python -m infer --model-name attn --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/attn_bce_balanced_final.pt
python -m infer --model-name attn --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/attn_bce_balanced_final_noisy.pt
python -m infer --model-name attn --loss-criterion bce --batch-size 50 --max-len 48000 --load-model output/states/attn_bce_unbalanced_final_cached.pt
python -m infer --model-name attn --loss-criterion crossentropy --batch-size 50 --max-len 48000 --load-model output/states/attn_crossentropy_balanced_final.pt
# bytenet networks
python -m infer --model-name bytenet --batch-size 10 --log-interval 20 --loss-criterion crossentropy --max-len 57344 --load-model output/states/bytenet_crossentropy_balanced_final.pt
