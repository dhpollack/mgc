from cfg import *
import time
from tqdm import tqdm
import json

if __name__ == '__main__':
    # python -m train --model-name squeezenet --data-path /mnt/data/mgc/data/audioset --batch-size 10 --use-cache
    msg = ""
    config = CFG()
    config.ds.target_transform = mgc_transforms.BinENC(config.ds.labels_dict)
    infer = config.test
    assert config.load_model  # we need to load a model
    with torch.no_grad():
        infer()
    #with open("output/losses_{}_{}_{}.json".format(config.model_name, config.loss_criterion, cur_epoch), "w") as f:
    #    losses = {
    #        "train_losses": config.train_losses,
    #        "valid_losses": config.valid_losses,
    #    }
    #    json.dump(losses, f)
