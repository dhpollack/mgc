from cfg import *
import time

if __name__ == '__main__':
    # python -m test.test_cfg --model-name squeezenet
    msg = ""
    config = CFG()
    train = config.fit
    save = config.save
    cur_epoch = config.cur_epoch
    for epoch in range(cur_epoch, 3):
        st = time.time()
        print("epoch {}{}".format(epoch + 1, msg))
        train(epoch, early_stop=25)
        if config.save_model and (epoch % config.chkpt_interval == 0 or epoch+1 == epochs):
            save(epoch)
        msg = " | previous epoch time: {0:.2f}s".format(time.time() - st)
