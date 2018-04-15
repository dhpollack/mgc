from cfg import *

if __name__ == '__main__':
    # python -m test.test_cfg --model-name squeezenet
    config = CFG()
    train = config.fit
    save = config.save
    cur_epoch = config.cur_epoch
    for epoch in range(cur_epoch, 1):
        print("epoch {}".format(epoch + 1))
        train(epoch, early_stop=None)
        if config.save_model and (epoch % config.chkpt_interval == 0 or epoch+1 == epochs):
            save(epoch)
