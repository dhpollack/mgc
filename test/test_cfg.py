from cfg import *
import time
from tqdm import tqdm

if __name__ == '__main__':
    # python -m test.test_cfg --model-name squeezenet --data-path /mnt/data/mgc/data/audioset --batch-size 10 --use-cache
    msg = ""
    EPOCHS = 30
    config = CFG()
    train = config.fit
    save = config.save
    cur_epoch = config.cur_epoch
    print(len(config.ds.data[config.ds.split]))
    with tqdm(range(cur_epoch, EPOCHS), total=EPOCHS, leave=True,
              postfix={"epoch": cur_epoch, "loss": "{0:.6f}".format(0.)}) as t:
        config.tqdmiter = t
        for epoch in t:
            st = time.time()
            train(epoch, early_stop=25)
            if config.save_model and (epoch % config.chkpt_interval == 0 or epoch+1 == epochs):
                save(epoch)
