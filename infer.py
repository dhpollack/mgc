from cfg import *
import time
from tqdm import tqdm
import json

if __name__ == '__main__':
    # python -m train --model-name squeezenet --data-path /mnt/data/mgc/data/audioset --batch-size 10 --use-cache
    msg = ""
    config = CFG()
    infer = config.test
    assert config.load_model  # we need to load a model
    infer()
    #with open("output/losses_{}_{}_{}.json".format(config.model_name, config.loss_criterion, cur_epoch), "w") as f:
    #    losses = {
    #        "train_losses": config.train_losses,
    #        "valid_losses": config.valid_losses,
    #    }
    #    json.dump(losses, f)
