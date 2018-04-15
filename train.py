from cfg import *
import time

if __name__ == '__main__':
    # python -m train --model-name squeezenet --data-path /mnt/data/mgc/data/audioset --batch-size 10 --use-cache
    msg = ""
    config = CFG()
    train = config.fit
    save = config.save
    cur_epoch = config.cur_epoch
    epochs = sum(config.epochs)
    print("total epochs: {}".format(epochs))
    for epoch in range(cur_epoch, epochs):
        st = time.time()
        print("epoch {}{}".format(epoch + 1, msg))
        train(epoch, early_stop=None)
        if config.save_model and (epoch % config.chkpt_interval == 0 or epoch+1 == epochs):
            save(epoch)
        msg = " | previous epoch time: {0:.2f}s".format(time.time() - st)
