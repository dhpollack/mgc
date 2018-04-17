from cfg import *
import time
from tqdm import tqdm

if __name__ == '__main__':
    # python -m train --model-name squeezenet --data-path /mnt/data/mgc/data/audioset --batch-size 10 --use-cache
    msg = ""
    config = CFG()
    train = config.fit
    save = config.save
    cur_epoch = config.cur_epoch
    epochs = sum(config.epochs)
    with tqdm(range(cur_epoch, epochs), total=epochs, initial=cur_epoch, leave=True,
              postfix={"epoch": cur_epoch, "loss": "{0:.6f}".format(0.)}) as t:
        config.tqdmiter = t
        for epoch in t:
            st = time.time()
            train(epoch, early_stop=None)
            if config.save_model and (epoch % config.chkpt_interval == 0 or epoch+1 == epochs):
                save(epoch)
