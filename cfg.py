import argparse
import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.autograd import Variable
import models
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import mgc_transforms
from loader_audioset import *
from tqdm import tqdm

class CFG(object):
    def __init__(self):
        parser = self.get_params()
        args = parser.parse_args()
        self.args = args
        self.save_model = args.save_model
        self.load_model = args.load_model
        self.chkpt_interval = args.chkpt_interval
        self.noises_dir = args.noises_dir
        self.use_precompute = args.use_precompute
        self.use_cache = args.use_cache
        self.data_path = args.data_path
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.log_interval = args.log_interval
        self.do_validate = args.validate
        self.max_len = args.max_len if args.max_len else 80000 # 160000 #10 secs
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.ngpu = torch.cuda.device_count()
        print("CUDA: {} with {} devices".format(self.use_cuda, self.ngpu))
        self.model_name = args.model_name
        self.loss_criterion = args.loss_criterion
        # load weights
        if args.load_model:
            state_dicts = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        else:
            state_dicts = {
                "models": None,
                "optimizer": None,
                "epoch": 0,
            }
        self.cur_epoch = state_dicts["epoch"]
        self.ds, self.dl = self.get_dataloader()
        self.model_list = self.get_models(state_dicts["models"])
        self.epochs, self.criterion, self.optimizer, self.scheduler = self.init_optimizer(state_dicts["optimizer"])
        if self.ngpu > 1 and "attn" not in self.model_name:
            self.model_list = [nn.DataParallel(m) for m in self.model_list]
        self.valid_losses = []
        self.train_losses = []
        self.tqdmiter = None

    def get_params(self):
        parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
        parser.add_argument('--data-path', type=str, default="data/audioset",
                            help='data path')
        parser.add_argument('--lr', type=float, default=0.0001,
                            help='initial learning rate')
        parser.add_argument('--epochs', type=int, default=10,
                            help='upper epoch limit')
        parser.add_argument('--batch-size', type=int, default=50, metavar='b',
                            help='batch size')
        parser.add_argument('--freq-bands', type=int, default=224,
                            help='number of frequency bands to use')
        parser.add_argument('--num-samples', type=int, default=None,
                            help='limit number of samples to load for testing')
        parser.add_argument('--dataset', type=str, default="balanced",
                            help='which Audioset dataset to use balanced / eval / unbalanced')
        parser.add_argument('--add-no-label', action='store_true',
                            help='add a label for "no label" or background noise')
        parser.add_argument('--use-cache', action='store_true',
                            help='use cache in the dataloader')
        parser.add_argument('--use-precompute', action='store_true',
                            help='precompute transformations')
        parser.add_argument('--noises-dir', type=str, default=None,
                            help='absolute path of noises to add to the audio')
        parser.add_argument('--num-workers', type=int, default=0,
                            help='number of workers for data loader')
        parser.add_argument('--validate', action='store_true',
                            help='do out-of-bag validation')
        parser.add_argument('--num-validate', type=int, default=None,
                            help='number of validation samples')
        parser.add_argument('--log-interval', type=int, default=5,
                            help='reports per epoch')
        parser.add_argument('--chkpt-interval', type=int, default=10,
                            help='how often to save checkpoints')
        parser.add_argument('--max-len', type=int, default=None,
                            help='max length of sample')
        parser.add_argument('--model-name', type=str, default="resnet34_conv",
                            help='name of model to use')
        parser.add_argument('--loss-criterion', type=str, default="bce",
                            help='loss criterion')
        parser.add_argument('--load-model', type=str, default=None,
                            help='path of model to load')
        parser.add_argument('--save-model', action='store_true',
                            help='path to save the final model')
        parser.add_argument('--train-full-model', action='store_true',
                            help='train full model vs. final layer')
        parser.add_argument('--seed', type=int, default=1111,
                            help='random seed')
        return parser


    def get_models(self, weights=None):
        NUM_CLASSES = len(self.ds.labels_dict)
        use_pretrained = True if not self.load_model else False
        if "resnet34" in self.model_name:
            model_list = [models.resnet.resnet34(use_pretrained, num_genres=NUM_CLASSES)]
        elif "resnet101" in self.model_name:
            model_list = [models.resnet.resnet101(use_pretrained, num_genres=NUM_CLASSES)]
        elif "squeezenet" in self.model_name:
            model_list = [models.squeezenet.squeezenet(use_pretrained, num_genres=NUM_CLASSES)]
        elif "attn" in self.model_name:
            self.hidden_size = 500
            kwargs_encoder = {
                "input_size": self.args.freq_bands // 2,
                "hidden_size": self.hidden_size,
                "n_layers": 1,
                "batch_size": self.batch_size
            }
            kwargs_decoder = {
                "input_size": self.args.freq_bands // 2,
                "hidden_size": self.hidden_size,
                "output_size": NUM_CLASSES,
                "attn_model": "general",
                "n_layers": 1,
                "dropout": 0.0, # was 0.1
                "batch_size": self.batch_size
            }
            model_list = models.attn.attn(kwargs_encoder, kwargs_decoder)
        elif "bytenet" in self.model_name:
            self.d = self.args.freq_bands // 2
            kwargs_encoder = {
                "d": self.d,
                "max_r": 16,
                "k": 3,
                "num_sets": 6,
                "reduce_out": [17, 3, 0, 2, 0, 7],
            }
            kwargs_decoder = {
                "d": self.d,
                "max_r": 16,
                "k": 3,
                "num_sets": 6,
                "num_classes": NUM_CLASSES,
                "reduce_out": None,
                "use_logsm": False,
            }
            model_list = models.bytenet.bytenet(kwargs_encoder, kwargs_decoder)
        # move model to GPU or multi-GPU
        model_list = [m.to(self.device) for m in model_list]
        # load weights
        if weights is not None:
            for i, sd in enumerate(weights):
                model_list[i].load_state_dict(sd)
        #if self.ngpu > 1:
        #    model_list = [nn.DataParallel(m) for m in model_list]
        return model_list

    def get_dataloader(self):
        usl = True if self.loss_criterion == "crossentropy" else False
        ds = AUDIOSET(self.data_path, dataset=self.args.dataset, noises_dir=self.noises_dir,
                      use_cache=False, num_samples=self.args.num_samples,
                      add_no_label=self.args.add_no_label, use_single_label=usl)
        if any(x in self.model_name for x in ["resnet34_conv", "resnet101_conv", "squeezenet"]):
            T = tat.Compose([
                    #tat.PadTrim(self.max_len, fill_value=1e-8),
                    mgc_transforms.SimpleTrim(self.max_len),
                    mgc_transforms.MEL(sr=16000, n_fft=600, hop_length=300, n_mels=self.args.freq_bands//2),
                    #mgc_transforms.Scale(),
                    mgc_transforms.BLC2CBL(),
                    mgc_transforms.Resize((self.args.freq_bands, self.args.freq_bands)),
                ])
        elif "_mfcc_librosa" in self.model_name:
            T = tat.Compose([
                    #tat.PadTrim(self.max_len, fill_value=1e-8),
                    mgc_transforms.SimpleTrim(self.max_len),
                    mgc_transforms.MFCC2(sr=16000, n_fft=600, hop_length=300, n_mfcc=12),
                    mgc_transforms.Scale(),
                    mgc_transforms.BLC2CBL(),
                    mgc_transforms.Resize((self.args.freq_bands, self.args.freq_bands)),
                ])
        elif "_mfcc" in self.model_name:
            sr = 16000
            ws = 800
            hs = ws // 2
            n_fft = 512 # 256
            n_filterbanks = 26
            n_coefficients = 12
            low_mel_freq = 0
            high_freq_mel = (2595 * math.log10(1 + (sr/2) / 700))
            mel_pts = torch.linspace(low_mel_freq, high_freq_mel, n_filterbanks + 2) # sr = 16000
            hz_pts = torch.floor(700 * (torch.pow(10,mel_pts / 2595) - 1))
            bins = torch.floor((n_fft + 1) * hz_pts / sr)
            td = {
                    "RfftPow": mgc_transforms.RfftPow(n_fft),
                    "FilterBanks": mgc_transforms.FilterBanks(n_filterbanks, bins),
                    "MFCC": mgc_transforms.MFCC(n_filterbanks, n_coefficients),
                 }

            T = tat.Compose([
                    #tat.PadTrim(self.max_len, fill_value=1e-8),
                    mgc_transforms.Preemphasis(),
                    mgc_transforms.SimpleTrim(self.max_len),
                    mgc_transforms.Sig2Features(ws, hs, td),
                    mgc_transforms.DummyDim(),
                    mgc_transforms.Scale(),
                    tat.BLC2CBL(),
                    mgc_transforms.Resize((self.args.freq_bands, self.args.freq_bands)),
                ])
        elif "attn" in self.model_name:
            T = tat.Compose([
                    mgc_transforms.SimpleTrim(self.max_len),
                    tat.MEL(sr=16000, n_fft=600, hop_length=300, n_mels=self.args.freq_bands//2),
                    mgc_transforms.Scale(),
                    mgc_transforms.SqueezeDim(2),
                    tat.LC2CL(),
                ])
        elif "bytenet" in self.model_name:
            offset = 714 # make clips divisible by 224
            T = tat.Compose([
                    mgc_transforms.SimpleTrim(self.max_len),
                    #tat.PadTrim(self.max_len - offset, fill_value=1e-8),
                    mgc_transforms.Scale(),
                    tat.LC2CL(),
                ])
        ds.transform = T
        if self.loss_criterion == "crossentropy":
            TT = mgc_transforms.XEntENC(ds.labels_dict)
            #TT = mgc_transforms.BinENC(ds.labels_dict, dtype=torch.int64)
        else:
            TT = mgc_transforms.BinENC(ds.labels_dict)
        ds.target_transform = TT
        ds.use_cache = self.use_cache
        if self.use_cache:
            ds.init_cache()
        if self.use_precompute:
            ds.load_precompute(self.model_name)
        dl = data.DataLoader(ds, batch_size=self.batch_size, drop_last=True,
                             num_workers=self.num_workers, collate_fn=bce_collate,
                             shuffle=True)
        if "attn" in self.model_name:
            dl.collate_fn = sort_collate
        return ds, dl

    def init_optimizer(self, weights=None):
        #if self.ngpu < 2 or "attn" in self.model_name:
        #    model = self.model
        #else:
        #    model = self.model.module
        model_list = self.model_list
        if self.loss_criterion == "softmargin":
            criterion = nn.MultiLabelSoftMarginLoss()
        elif self.loss_criterion == "margin":
            criterion = nn.MultiLabelMarginLoss()
        elif self.loss_criterion == "crossentropy":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        epochs = None
        if "squeezenet" in self.model_name:
            if self.dataset == "unbalanced":
                epochs = [5, 10, 35, 50]
            else:
                epochs = [10, 20, 100]
            opt_type = torch.optim.Adam
            opt_params = [
                {"params": model_list[0][1].features.parameters(), "lr": 0.},
                {"params": model_list[0][1].classifier.parameters(), "lr": self.args.lr}
            ]
            opt_kwargs = {"amsgrad": True}
        elif any(x in self.model_name for x in ["resnet34", "resnet101"]):
            if "resnet34" in self.model_name:
                if self.dataset == "unbalanced":
                    epochs = [10, 25, 40, 50]
                else:
                    epochs = [20, 60, 100, 120]
            elif "resnet101" in self.model_name:
                if self.dataset == "unbalanced":
                    epochs = [10, 20, 28, 33]
                else:
                    epochs = [20, 40, 80]
            opt_type = torch.optim.Adam
            feature_params = nn.ParameterList()
            for m in list(model_list[0][1].children())[:-1]:
                feature_params.extend(m.parameters())
            fc_params = model_list[0][1].fc.parameters()
            opt_params = [
                {"params": feature_params, "lr": 0.}, # features
                {"params": fc_params, "lr": self.args.lr} # classifier
            ]
            opt_kwargs = {"amsgrad": True}
        elif any(x in self.model_name for x in ["attn", "bytenet"]):
            if self.dataset == "unbalanced":
                epochs = [25, 40]
            else:
                epochs = [70, 100]
            opt_type = torch.optim.SGD
            opt_params = [
                {"params": model_list[0].parameters(), "lr": self.args.lr},
                {"params": model_list[1].parameters(), "lr": self.args.lr}
            ]
            opt_kwargs = {"momentum": 0.9}
        optimizer = opt_type(opt_params, **opt_kwargs)
        if weights is not None:
            optimizer.load_state_dict(weights)
            # https://github.com/pytorch/pytorch/issues/2830, fixed in master?
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        v = v.to(self.device)
                        state[k] = v
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epochs[1:], gamma=0.4)

        return epochs, criterion, optimizer, scheduler

    def adjust_opt_params(self, epoch):
        # automate fine tuning
        if epoch == self.epochs[0]:
            if "squeezenet" in self.model_name:
                self.optimizer.param_groups[0]["initial_lr"] = self.scheduler.base_lrs[0] = self.optimizer.param_groups[1]["initial_lr"]
                self.optimizer.param_groups[0]["lr"] = self.args.lr
            elif "resnet" in self.model_name:
                self.optimizer.param_groups[0]["initial_lr"] = self.scheduler.base_lrs[0] = self.optimizer.param_groups[1]["initial_lr"]
                self.optimizer.param_groups[0]["lr"] = self.args.lr
            elif "attn" in self.model_name:
                # no finetuning of these models yet
                pass
            elif "bytenet" in self.model_name:
                # no finetuning of these models yet
                pass
            #print(self.optimizer)

    def fit(self, epoch, early_stop=None):
        epoch_losses = []
        self.ds.set_split("train")
        self.adjust_opt_params(epoch)
        self.scheduler.step()
        #self.optimizer = self.get_optimizer(epoch)
        num_batches = len(self.dl)
        if any(x in self.model_name for x in ["resnet", "squeezenet"]):
            if self.use_precompute:
                pass # TODO implement network precomputation
                #self.precompute(self.L["fc_layer"]["precompute"])
            m = self.model_list[0]
            with tqdm(total=num_batches, leave=False, position=1) as t:
                for i, (mb, tgts) in enumerate(self.dl):
                    if i == early_stop: break
                    m.train()
                    mb, tgts = mb.to(self.device), tgts.to(self.device)
                    m.zero_grad()
                    out = m(mb)
                    if "margin" in self.loss_criterion:
                        out = F.sigmoid(out)
                    if self.loss_criterion == "margin":
                        tgts = tgts.long()
                    #print(tgts)
                    loss = self.criterion(out, tgts)
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.item())
                    if self.tqdmiter:
                        self.tqdmiter.set_postfix({"epoch": epoch, "loss": "{0:.6f}".format(epoch_losses[-1])})
                        self.tqdmiter.refresh()
                    else:
                        print(epoch_losses[-1])
                    if i % self.log_interval == 0 and self.do_validate and i != 0:
                        with torch.no_grad():
                            self.validate(epoch)
                            self.ds.set_split("train")
                    t.update()
        elif "attn" in self.model_name:
            encoder = self.model_list[0]
            decoder = self.model_list[1]
            with tqdm(total=num_batches, leave=False, position=1) as t:
                for i, ((mb, lengths), tgts) in enumerate(self.dl):
                    # set model into train mode and clear gradients
                    encoder.train()
                    decoder.train()
                    encoder.zero_grad()
                    decoder.zero_grad()

                    # set inputs and targets
                    mb, tgts = mb.to(self.device), tgts.to(self.device)

                    # create the initial hidden input before packing sequence
                    encoder_hidden = encoder.initHidden(mb)

                    # pack sequence
                    mb = pack(mb, lengths, batch_first=True)
                    #print(mb.size(), tgts.size())
                    # encode sequence
                    encoder_output, encoder_hidden = encoder(mb, encoder_hidden)

                    # Prepare input and output variables for decoder
                    #dec_size = [[[0] * encoder.hidden_size]*1]*self.batch_size
                    #print(encoder_output.detach().new(dec_size).size())
                    #enc_out_var, enc_out_len = unpack(encoder_output, batch_first=True)
                    #dec_i = enc_out_var.new_zeros((self.batch_size, 1, encoder.hidden_size))
                    dec_h = encoder_hidden # Use last (forward) hidden state from encoder
                    #print(decoder.n_layers, encoder_hidden.size(), dec_i.size(), dec_h.size())

                    # run through decoder in one shot
                    mb, _ = unpack(mb, batch_first=True)
                    dec_o, dec_h, dec_attn = decoder(mb, dec_h, encoder_output)
                    dec_o.squeeze_()
                    #print(dec_o)
                    #print(dec_o.size(), dec_h.size(), dec_attn.size(), tgts.size())
                    #print(dec_o.view(-1, decoder.output_size).size(), tgts.view(-1).size())

                    # calculate loss and backprop
                    if "margin" in self.loss_criterion:
                        dec_o = F.sigmoid(dec_o)
                    if self.loss_criterion == "margin":
                        tgts = tgts.long()
                    loss = self.criterion(dec_o, tgts)
                    #nn.utils.clip_grad_norm(encoder.parameters(), 0.05)
                    #nn.utils.clip_grad_norm(decoder.parameters(), 0.05)
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.item())
                    if self.tqdmiter:
                        self.tqdmiter.set_postfix({"epoch": epoch, "loss": "{0:.6f}".format(epoch_losses[-1])})
                        self.tqdmiter.refresh()
                    else:
                        print(epoch_losses[-1])
                    if i % self.log_interval == 0 and self.do_validate and i != 0:
                        with torch.no_grad():
                            self.validate(epoch)
                            self.ds.set_split("train")
                    t.update()
        elif "bytenet" in self.model_name:
            encoder = self.model_list[0]
            decoder = self.model_list[1]
            with tqdm(total=num_batches, leave=False, position=1) as t:
                for i, (mb, tgts) in enumerate(self.dl):
                    # set model into train mode and clear gradients
                    encoder.train()
                    decoder.train()
                    encoder.zero_grad()
                    decoder.zero_grad()
                    # set inputs and targets
                    mb, tgts = mb.to(self.device), tgts.to(self.device)
                    mb = encoder(mb)
                    mb.unsqueeze_(1)
                    out = decoder(mb)
                    if "margin" in self.loss_criterion:
                        out = F.sigmoid(out)
                    if self.loss_criterion == "margin":
                        tgts = tgts.long()
                    loss = self.criterion(out, tgts)
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.item())
                    if self.tqdmiter:
                        self.tqdmiter.set_postfix({"epoch": epoch, "loss": "{0:.6f}".format(epoch_losses[-1])})
                        self.tqdmiter.refresh()
                    else:
                        print(epoch_losses[-1])
                    if i % self.log_interval == 0 and self.do_validate and i != 0:
                        with torch.no_grad():
                            self.validate(epoch)
                            self.ds.set_split("train")
                    t.update()
        self.train_losses.append(epoch_losses)
        if epoch % self.args.chkpt_interval == 0 and epoch != 0:
            self.ds.init_cache()

    def validate(self, epoch):
        self.ds.set_split("valid", self.args.num_samples)
        running_validation_loss = []
        accuracies = []
        acc = 0
        threshold = 1 - (1. / 3.)
        num_batches = len(self.dl)
        if any(x in self.model_name for x in ["resnet", "squeezenet"]):
            m = self.model_list[0]
            # set model(s) into eval mode
            m.eval()
            with tqdm(total=num_batches, leave=True, position=2,
                      postfix={"acc": acc, "loss": "{0:.6f}".format(0.)}) as t:
                for mb_valid, tgts_valid in self.dl:
                    mb_valid = mb_valid.to(self.device)
                    tgts_valid = tgts_valid.to(torch.device("cpu"))
                    out_valid = m(mb_valid)
                    out_valid = out_valid.to(torch.device("cpu"))
                    if "margin" in self.loss_criterion:
                        out_valid = F.sigmoid(out_valid)
                    if self.loss_criterion == "margin":
                        tgts_valid = tgts_valid.long()
                    loss_valid = self.criterion(out_valid, tgts_valid)
                    running_validation_loss += [loss_valid.item()]
                    if "margin" not in self.loss_criterion:
                        out_valid = F.sigmoid(out_valid)
                    if self.loss_criterion == "crossentropy":
                        out_pred = out_valid.max(1)[1]
                        acc = (out_pred == tgts_valid).sum().item() / tgts_valid.size(0)
                    else:
                        out_mask = out_valid > threshold
                        acc = np.logical_and(out_mask.numpy()==True, tgts_valid.numpy()==True).sum() / (tgts_valid.numpy()==True).sum()
                    accuracies.append(acc)
                    t.set_postfix({"acc": acc, "loss": "{0:.6f}".format(running_validation_loss[-1])})
                    t.update()
                    #correct += (out_valid.detach().max(1)[1] == tgts_valid.detach()).sum()
        elif "attn" in self.model_name:
            encoder = self.model_list[0]
            decoder = self.model_list[1]
            # set model(s) into eval mode
            encoder.eval()
            decoder.eval()
            with tqdm(total=num_batches, leave=False, position=2,
                      postfix={"acc": acc, "loss": "{0:.6f}".format(0.)}) as t:
                for i, ((mb_valid, lengths), tgts_valid) in enumerate(self.dl):
                    # set model into train mode and clear gradients

                    # move inputs to cuda if required
                    mb_valid = mb_valid.to(self.device)
                    tgts_valid = tgts_valid.to(torch.device("cpu"))
                    # init hidden before packing
                    encoder_hidden = encoder.initHidden(mb_valid)

                    # set inputs and targets
                    mb_valid = pack(mb_valid, lengths, batch_first=True)
                    #print(mb.size(), tgts.size())
                    encoder_output, encoder_hidden = encoder(mb_valid, encoder_hidden)

                    #print(encoder_output.detach().new(dec_size).size())
                    #enc_out_var, enc_out_len = unpack(encoder_output, batch_first=True)
                    #dec_i = enc_out_var.new_zeros((self.batch_size, 1, encoder.hidden_size))
                    dec_h = encoder_hidden # Use last (forward) hidden state from encoder
                    #print(decoder.n_layers, encoder_hidden.size(), dec_i.size(), dec_h.size())

                    # run through decoder in one shot
                    mb_valid, _ = unpack(mb_valid, batch_first=True)
                    out_valid, dec_h, dec_attn = decoder(mb_valid, dec_h, encoder_output)
                    # calculate loss
                    out_valid = out_valid.to(torch.device("cpu"))
                    out_valid.squeeze_()
                    if "margin" in self.loss_criterion:
                        out_valid = F.sigmoid(out_valid)
                    if self.loss_criterion == "margin":
                        tgts_valid = tgts_valid.long()
                    loss_valid = self.criterion(out_valid, tgts_valid)
                    running_validation_loss += [loss_valid.item()]
                    if "margin" not in self.loss_criterion:
                        out_valid = F.sigmoid(out_valid)
                    if self.loss_criterion == "crossentropy":
                        out_pred = out_valid.max(1)[1]
                        acc = (out_pred == tgts_valid).sum().item() / tgts_valid.size(0)
                    else:
                        out_mask = out_valid > threshold
                        acc = np.logical_and(out_mask.numpy()==True, tgts_valid.numpy()==True).sum() / (tgts_valid.numpy()==True).sum()
                    accuracies.append(acc)
                    t.set_postfix({"acc": acc, "loss": "{0:.6f}".format(running_validation_loss[-1])})
                    t.update()
                    #correct += (dec_o.detach().max(1)[1] == tgts.detach()).sum()
        elif "bytenet" in self.model_name:
            encoder = self.model_list[0]
            decoder = self.model_list[1]
            # set model(s) into eval mode
            encoder.eval()
            decoder.eval()
            with tqdm(total=num_batches, leave=False, position=2,
                      postfix={"acc": acc, "loss": "{0:.6f}".format(0.)}) as t:
                for i, (mb_valid, tgts_valid) in enumerate(self.dl):
                    # set inputs and targets
                    mb_valid, tgts_valid = mb_valid.to(self.device), tgts_valid.to(torch.device("cpu"))
                    mb_valid = encoder(mb_valid)
                    # turn 3d input into 4d input for classifier
                    mb_valid.unsqueeze_(1)
                    out_valid = decoder(mb_valid)
                    if "margin" in self.loss_criterion:
                        out_valid = F.sigmoid(out_valid)
                    if self.loss_criterion == "margin":
                        tgts_valid = tgts_valid.long()
                    out_valid = out_valid.to(torch.device("cpu"))
                    loss_valid = self.criterion(out_valid, tgts_valid)
                    running_validation_loss += [loss_valid.item()]
                    if "margin" not in self.loss_criterion:
                        out_valid = F.sigmoid(out_valid)
                    if self.loss_criterion == "crossentropy":
                        out_pred = out_valid.max(1)[1]
                        acc = (out_pred == tgts_valid).sum().item() / tgts_valid.size(0)
                    else:
                        out_mask = out_valid > threshold
                        acc = np.logical_and(out_mask.numpy()==True, tgts_valid.numpy()==True).sum() / (tgts_valid.numpy()==True).sum()
                    accuracies.append(acc)
                    t.set_postfix({"acc": acc, "loss": "{0:.6f}".format(running_validation_loss[-1])})
                    t.update()
                    #correct += (dec_o.detach().max(1)[1] == tgts.detach()).sum()

        self.valid_losses.append((running_validation_loss, accuracies))

    def test(self):
        self.ds.set_split("test", self.args.num_samples)
        thresh = 1. / 50.
        acc = 0.
        num_batches = len(self.dl)
        num_labels = len(self.ds.labels_dict)
        jsondata = []
        counter_array = np.zeros((num_labels, 6)) # tgts, preds, tp, fp, tn, fn
        if any(x in self.model_name for x in ["resnet", "squeezenet"]):
            m = self.model_list[0]
            # set model(s) into eval mode
            m.eval()
            with tqdm(total=num_batches, leave=False, position=1,
                      postfix={"acc": acc}) as t:
                for mb, tgts in self.dl:
                    mb = mb.to(self.device)
                    tgts = tgts.to(torch.device("cpu"))
                    # run inference
                    out = m(mb)
                    # move output to cpu for analysis / numpy
                    out = out.to(torch.device("cpu"))
                    jsondata.append((out.numpy().tolist(), tgts.numpy().tolist()))
                    if self.loss_criterion == "crossentropy":
                        out = F.softmax(out, dim = 1)
                    else:
                        out = F.sigmoid(out)
                        #out = F.softmax(out, dim = 1)
                    # out is either size (N, C) or (N, )
                    for tgt, o in zip(tgts, out):
                        tgt = tgt.numpy()
                        tgt_mask = tgt == 1.
                        counter_array[tgt_mask, 0] += 1
                        o_mask = o >= thresh
                        o_mask = o_mask.numpy()
                        o_mask = o_mask.astype(np.bool)
                        #print(o_mask); break;

                        counter_array[o_mask, 1] += 1
                        tp = np.logical_and(tgt_mask==True, o_mask==True)  # this will be deflated for cross entorpy
                        fp = np.logical_and(tgt_mask==False, o_mask==True)
                        tn = np.logical_and(tgt_mask==False, o_mask==False)
                        fn = np.logical_and(tgt_mask==True, o_mask==False)

                        counter_array[tp, 2] += 1
                        counter_array[fp, 3] += 1
                        counter_array[tn, 4] += 1
                        counter_array[fn, 5] += 1

                        k = int(np.sum(tgt_mask))
                        tmp1 = torch.topk(o, k)[1]  # get indicies
                        tmp2 = np.where(tgt == 1.)[0]
                    #acc = counter_array[:, 0].sum() / counter_array[:, 0].sum()
                    #t.set_postfix({"acc": acc, "loss": "{0:.6f}".format(last_five_ave)})
                    t.update()
                    #correct += (out_valid.detach().max(1)[1] == tgts_valid.detach()).sum()

        else:
            raise NotImplemented
        #json.dump(jsondata, open("output/tmp/test_output.json", "w"))
        print(counter_array.astype(np.int))
    def get_train(self):
        return self.fit

    def save(self, epoch):
        mstate = {
            "models": [m.module.state_dict() if isinstance(m, nn.DataParallel) else m.state_dict() for m in self.model_list],
            "optimizer": self.optimizer.module.state_dict() if isinstance(self.optimizer, nn.DataParallel) else self.optimizer.state_dict(),
            "epoch": epoch+1,
        }
        is_noisy = "_noisy" if self.noises_dir else ""
        sname = "output/states/{}{}_{}_{}.pt".format(self.model_name, is_noisy, self.loss_criterion, epoch+1)
        torch.save(mstate, sname)

    def precompute(self, m):
        if "resnet" in self.model_name:
            dl = data.DataLoader(self.ds, batch_size=self.batch_size,
                                 num_workers=self.num_workers, shuffle=False)
            m.eval()
            for splt in ["train", "valid"]:
                self.ds.set_split(splt)
                c = self.ds.splits[splt].start
                for i, (mb, tgts) in enumerate(dl):
                    bs = mb.size(0)
                    mb = mb.to(self.device)
                    m.zero_grad()
                    out = m(mb).to(torch.device("cpu"))
                    out = out.detach()
                    for j_i, j_k in enumerate(range(c, c+bs)):
                        idx_split = self.ds.splits[splt][j_k]
                        k = self.ds.detach()[idx_split]
                        self.ds.cache[k] = (out[j_i], tgts[j_i])
                    c += bs

def sort_collate(batch):
    """Sorts data and lengths by length of data then returns
       the (padded data, lengths) and labels.

       Args:
         batch: (list of tuples) [(sig, label)].
             sig is a FloatTensor
             label is an int
       Output:
         sigs: (FloatTensor), padded signals in desc lenght order
         lengths: (list(int)), list of original lengths of sigs
         labels: (LongTensor), labels from the file names of the wav.

    """
    if len(batch) == 1:
        sigs, labels = batch[0][0], batch[0][1]
        lengths = [sigs.size(0)]
        #sigs = sigs.t()
        sigs.unsqueeze_(0)
        labels = tensor.LongTensor([labels]).unsqueeze(0)
    if len(batch) > 1:
        sigs, labels, lengths = zip(*[(a, b, a.size(0)) for (a,b) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
        max_len, n_feats = sigs[0].size()
        sigs = [pad_sig(s, max_len, n_feats) if s.size(0) != max_len else s for s in sigs]
        sigs = torch.stack(sigs, 0)
        lengths = np.array(lengths)
        labels = torch.stack(labels, 0)
    return (sigs, lengths), labels

def pad_sig(s, max_len, n_feats):
    s_len = s.size(0)
    s_new = s.new(max_len, n_feats).fill_(0)
    s_new[:s_len] = s
    return s_new
