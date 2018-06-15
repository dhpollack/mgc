from cfg import *
from tqdm import tqdm
import json

if __name__ == '__main__':
    # python -m infer --model-name squeezenet --batch-size 10 --load-model output/states/resnet34_conv_crossentropy_140.pt
    config = CFG()
    config.ds.target_transform = mgc_transforms.BinENC(config.ds.labels_dict)
    infer = config.test
    assert config.load_model  # we need to load a model
    with torch.no_grad():
        infer()
    cached = "cache" if config.use_cache else "nocache"
    noise = "noise" if config.noises_dir else "nonoise"
    json_name = "output/inference_{}_{}_{}_{}_{}.json".format(config.model_name, config.loss_criterion, config.dataset, noise, cached)
    json.dump(config.infer_stats.tolist(), open(json_name, "w"))
    json_name = "output/output_{}_{}_{}_{}_{}.json".format(config.model_name, config.loss_criterion, config.dataset, noise, cached)
    json.dump(config.infer_outputs, open(json_name, "w"))
