import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.autograd import Variable
import models
import torchaudio.transforms as tat
import torchvision.transforms as tvt
import mgc_transforms
from loader_audioset import *
import math

parser = argparse.ArgumentParser(description='PyTorch Language ID Classifier Trainer')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=100, metavar='b',
                    help='batch size')
parser.add_argument('--freq-bands', type=int, default=224,
                    help='number of frequency bands to use')
parser.add_argument('--data-path', type=str, default="data/audioset/balanced",
                    help='data path')
parser.add_argument('--use-cache', action='store_true',
                    help='use cache in the dataloader')
parser.add_argument('--use-precompute', action='store_true',
                    help='precompute transformations')
parser.add_argument('--mixin-noise', action='store_true',
                    help='precompute transformations')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers for data loader')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--chkpt-interval', type=int, default=10,
                    help='how often to save checkpoints')
parser.add_argument('--model-name', type=str, default="resnet34_conv",
                    help='data path')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
parser.add_argument('--train-full-model', action='store_true',
                    help='train full model vs. final layer')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
args = parser.parse_args()


class CFG(object):
    def __init__(self):
        self.max_len = 150000
        self.use_cuda = torch.cuda.is_available()
        self.ngpu = torch.cuda.device_count()
        print("CUDA: {} with {} devices".format(self.use_cuda, self.ngpu))
        self.model_name = args.model_name
        # load weights
        if args.load_model:
            state_dicts = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        else:
            state_dicts = {
                "models": None,
                "optimizers": None,
                "epoch": 0,
            }
        self.model_list = self.get_model(state_dicts["models"])
        self.ds, self.dl = self.get_dataloader()
        self.epochs, self.criterion, self.optimizers = self.init_optimizer(state_dicts["optimizers"])
        if self.ngpu > 1:
            self.model_list = [nn.DataParallel(m).cuda() for m in self.model_list]
        self.save_model = args.save_model
        self.chkpt_interval = args.chkpt_interval
        self.mixin_noise = args.mixin_noise
        self.valid_losses = []
        self.train_losses = []

    def get_models(self, weights=None):
        if "resnet34" in self.model_name:
            model_list = [models.resnet.resnet34(use_pretrained, num_langs=5)]
        elif "resnet101" in self.model_name:
            model_list = [models.resnet.resnet101(use_pretrained, num_langs=5)]
        elif "squeezenet" in self.model_name:
            model_list = [models.squeezenet.squeezenet(use_pretrained, num_langs=5)]
        elif "attn" in self.model_name:
            self.hidden_size = 500
            kwargs_encoder = {
                "input_size": args.freq_bands,
                "hidden_size": self.hidden_size,
                "n_layers": 1,
                "batch_size": args.batch_size
            }
            kwargs_decoder = {
                "hidden_size": self.hidden_size,
                "output_size": 5,
                "attn_model": "general",
                "n_layers": 1,
                "dropout": 0.0, # was 0.1
                "batch_size": args.batch_size
            }
            model_list = models.attn.attn(kwargs_encoder, kwargs_decoder)
        elif "bytenet" in self.model_name:
            self.d = 800
            kwargs_encoder = {
                "d": self.d,
                "max_r": 16,
                "k": 3,
                "num_sets": 6,
                "reduce_out": [0, 4, 4, 4, 4, 2],
            }
            kwargs_decoder = {
                "d": self.d,
                "max_r": 16,
                "k": 3,
                "num_sets": 6,
                "num_classes": 5,
                "reduce_out": None,
                "use_logsm": False,
            }
            model_list = models.bytenet.bytenet(kwargs_encoder, kwargs_decoder)
        # move model to GPU or multi-GPU
        if self.use_cuda:
            model_list = [m.cuda() for m in model_list]
        # load weights
        if weights is not None:
            for i, sd in enumerate(weights):
                model_list[i].load_state_dict(sd)
        #if self.ngpu > 1:
        #    model_list = [nn.DataParallel(m).cuda() for m in model_list]
        return model_list

    def get_dataloader(self):
        ds = AUDIOSET(args.data_path, mix_noise=args.mixin_noise,
                      use_cache=args.use_cache, use_precompute=args.use_precompute)
        if any((x for x in ["resnet34_conv", "resnet101_conv", "squeezenet"] if x == self.model_name)):
            T = tat.Compose([
                    #tat.PadTrim(self.max_len),
                    tat.MEL(n_mels=224),
                    tat.BLC2CBL(),
                    tvt.ToPILImage(),
                    tvt.Resize((224, 224)),
                    tvt.ToTensor(),
                ])
        elif self.model_name == "resnet34_mfcc":
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
                    tat.Scale(),
                    #tat.PadTrim(self.max_len, fill_value=1e-8),
                    mgc_transforms.Preemphasis(),
                    mgc_transforms.Sig2Features(ws, hs, td),
                    mgc_transforms.DummyDim(),
                    tat.BLC2CBL(),
                    tvt.ToPILImage(),
                    tvt.Resize((224, 224)),
                    tvt.ToTensor(),
                ])
        elif "attn" in self.model_name:
            T = tat.Compose([
                    tat.MEL(n_mels=224),
                    mgc_transforms.SqueezeDim(2),
                    tat.LC2CL(),
                    #tat.BLC2CBL(),
                ])
        elif "bytenet" in self.model_name:
            T = tat.Compose([
                    tat.LC2CL(),
                ])
        TT = mgc_transforms.BinENC(ds.labels_dict)
        ds.transform = T
        ds.target_transform = TT
        if args.use_precompute:
            ds.load_precompute(args.model_name)
        dl = data.DataLoader(ds, batch_size=args.batch_size, drop_last=True,
                             num_workers=args.num_workers, collate_fn=bce_collate,
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
        optimizers = []
        criterion = None
        epochs = None
        if any((x for x in ["resnet34", "resnet101", "squeezenet"] if x == self.model_name)):
            criterion = nn.CrossEntropyLoss()
            if "resnet34" in self.model_name:
                epochs = [40, 100]
            elif "resnet101" in self.model_name:
                epochs = [20, 50]
            elif "squeezenet" in self.model_name:
                epochs = [100]
            if any((x for x in ["resnet34", "resnet101"] if x == self.model_name)):
                optimizer_fc = torch.optim.Adam
                optimizer_fc_params = model_list[0][1].fc.parameters()
                optimizer_fc_kwargs = {"lr": 0.0001,}
                optimizer_fc_precom = nn.Sequential(model_list[0][0],
                                                    *list(model_list[0][1].children())[:-1])
                optimzers.append(optimizer_fc(optimizer_fc_params, **optimizer_fc_kwargs))
            optimizer_full = torch.optim.SGD
            optimizer_full_params = model_list[0].parameters()
            optimizer_full_kwargs = {"lr": 0.0001, "momentum": 0.9,}
            optimizers.append(optimizer_full(optimizer_full_params, **optimizer_full_kwargs))
        elif any((x for x in ["attn", "bytenet"] if x == self.model_name)):
            criterion = nn.CrossEntropyLoss()
            epochs = [100]
            if "attn" in self.model_name:
                opt = torch.optim.RMSprop
            elif "bytenet" in self.model_name:
                opt = torch.optim.SGD
            opt_params = [
                    {"params": model_list[0].parameters()},
                    {"params": model_list[1].parameters()}
                ]
            opt_kwargs = {"lr": 0.0001, "momentum": 0.9,}
            optimizers.append(opt(opt_params, **opt_kwargs))
        if weights is not None:
            for i, sd in enumerate(weights):
                optimizers[i].load_state_dict(sd)
        return epochs, criterion, optimizers

    def get_optimizer(self, epoch):
        grenz = 0
        for i, v in enumerate(self.epochs):
            grenz += v
            if epoch == 0:
                return self.optimizers[0]
            elif epoch == grenz:
                print("Using new optimizer: {}".format(self.optimizers[i+1]))
                return self.optimizers[i+1]
            else:
                pass
        return self.optimizer

    def fit(self, epoch):
        if any((x for x in ["resnet", "squeezenet"] if x in self.model_name)):
            if args.use_precompute:
                pass # TODO implement network precomputation
                #self.precompute(self.L["fc_layer"]["precompute"])
            self.ds.set_split("train")
            self.optimizer = self.get_optimizer(epoch)
            epoch_losses = []
            m = self.model_list[0]
            for i, (mb, tgts) in enumerate(self.dl):
                m.train()
                if self.use_cuda:
                    mb, tgts = mb.cuda(), tgts.cuda()
                mb, tgts = Variable(mb), Variable(tgts)
                m.zero_grad()
                out = m(mb)
                loss = self.criterion(out, tgts)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.data[0])
                print(loss.data[0])
                if i % args.log_interval == 0 and args.validate and i != 0:
                    self.validate(epoch)
                self.ds.set_split("train")
            self.train_losses.append(epoch_losses)
        if "attn" in self.model_name:
            self.ds.set_split("train")
            self.optimizer = self.get_optimizer(epoch)
            epoch_losses = []
            encoder = self.model_list[0]
            decoder = self.model_list[1]
            input_type = torch.FloatTensor
            for i, ((mb, lengths), tgts) in enumerate(self.dl):
                # set model into train mode and clear gradients
                encoder.train()
                decoder.train()
                encoder.zero_grad()
                decoder.zero_grad()

                # set inputs and targets
                if self.use_cuda:
                    mb, tgts = mb.cuda(), tgts.cuda()
                mb = pack(Variable(mb), lengths, batch_first=True)
                tgts = Variable(tgts)
                #print(mb.size(), tgts.size())
                encoder_hidden = encoder.initHidden(input_type)
                encoder_output, encoder_hidden = encoder(mb, encoder_hidden)

                # Prepare input and output variables for decoder
                dec_size = [[[0] * encoder.hidden_size]*1]*args.batch_size
                #print(encoder_output.data.new(dec_size).size())
                enc_out_var, enc_out_len = unpack(encoder_output, batch_first=True)
                dec_i = Variable(enc_out_var.data.new(dec_size))
                #dec_i = Variable(encoder_output.data.new(dec_size))
                dec_h = encoder_hidden # Use last (forward) hidden state from encoder
                #print(decoder.n_layers, encoder_hidden.size(), dec_i.size(), dec_h.size())

                # run through decoder in one shot
                dec_o, dec_h, dec_attn = decoder(dec_i, dec_h, encoder_output)
                #print(dec_o)
                #print(dec_o.size(), dec_h.size(), dec_attn.size())
                #print(dec_o.view(-1, decoder.output_size).size(), tgts.view(-1).size())

                # calculate loss and backprop
                loss = self.criterion(dec_o.view(-1, decoder.output_size), tgts.view(-1))
                #nn.utils.clip_grad_norm(encoder.parameters(), 0.05)
                #nn.utils.clip_grad_norm(decoder.parameters(), 0.05)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.data[0])
                print(loss.data[0])
                if i % args.log_interval == 0 and args.validate and i != 0:
                    self.validate(epoch)
                self.ds.set_split("train")
                self.train_losses.append(epoch_losses)
        if "bytenet" in self.model_name:
            self.ds.set_split("train")
            self.optimizer = self.get_optimizer(epoch)
            epoch_losses = []
            encoder = self.model[0]
            decoder = self.model[1]
            for i, (mb, tgts) in enumerate(self.dl):
                # set model into train mode and clear gradients
                encoder.train()
                decoder.train()
                encoder.zero_grad()
                decoder.zero_grad()
                # set inputs and targets
                if self.use_cuda:
                    mb, tgts = mb.cuda(), tgts.cuda()
                mb, tgts = Variable(mb), Variable(tgts)
                mb = encoder(mb)
                out = decoder(mb)
                loss = criterion(out, tgts) # ach, alles für Bilder. fixed in master for 0.3.0 use (out.unsqueeze(2), tgts.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.data[0])
                print(loss.data[0])
                if i % args.log_interval == 0 and args.validate and i != 0:
                    self.validate(epoch)
                self.ds.set_split("train")
                self.train_losses.append(epoch_losses)


    def validate(self, epoch):
        if any((x for x in ["resnet", "squeezenet"] if x in self.model_name)):
            m = self.model_list[0]
            m.eval()
            self.ds.set_split("valid")
            running_validation_loss = 0
            correct = 0
            num_batches = len(self.dl)
            for mb_valid, tgts_valid in self.dl:
                if self.use_cuda:
                    mb_valid, tgts_valid = mb_valid.cuda(), tgts_valid.cuda()
                mb_valid, tgts_valid = Variable(mb_valid), Variable(tgts_valid)
                out_valid = m(mb_valid)
                out_valid, tgts_valid = out_valid.cpu(), tgts_valid.cpu()
                loss_valid = self.criterion(out_valid, tgts_valid)
                running_validation_loss += loss_valid.data[0]
                correct += (out_valid.data.max(1)[1] == tgts_valid.data).sum()
        elif "attn" in self.model_name:
            self.ds.set_split("valid")
            running_validation_loss = 0
            correct = 0
            num_batches = len(self.dl)
            encoder = self.model_list[0]
            decoder = self.model_list[1]
            input_type = torch.FloatTensor
            for i, ((mb, lengths), tgts) in enumerate(self.dl):
                # set model into train mode and clear gradients
                encoder.eval()
                decoder.eval()

                # set inputs and targets
                if self.use_cuda:
                    mb, tgts = mb.cuda(), tgts.cuda()
                mb = pack(Variable(mb), lengths, batch_first=True)
                tgts = Variable(tgts)
                #print(mb.size(), tgts.size())
                encoder_hidden = encoder.initHidden(input_type)
                encoder_output, encoder_hidden = encoder(mb, encoder_hidden)

                # Prepare input and output variables for decoder
                dec_size = [[[0] * encoder.hidden_size]*1]*args.batch_size
                #print(encoder_output.data.new(dec_size).size())
                enc_out_var, enc_out_len = unpack(encoder_output, batch_first=True)
                dec_i = Variable(enc_out_var.data.new(dec_size))
                #dec_i = Variable(encoder_output.data.new(dec_size))
                dec_h = encoder_hidden # Use last (forward) hidden state from encoder
                #print(decoder.n_layers, encoder_hidden.size(), dec_i.size(), dec_h.size())

                # run through decoder in one shot
                dec_o, dec_h, dec_attn = decoder(dec_i, dec_h, encoder_output)
                # calculate loss and backprop
                dec_o, tgts = dec_o.cpu(), tgts.cpu()
                dec_o = dec_o.view(-1, decoder.output_size)
                loss = self.criterion(dec_o, tgts.view(-1))
                running_validation_loss += loss.data[0]
                correct += (dec_o.data.max(1)[1] == tgts.data).sum()

        self.valid_losses.append((running_validation_loss / num_batches, correct / len(self.ds)))
        print("loss: {}, acc: {}".format(running_validation_loss / num_batches, correct / len(self.ds)))

    def get_train(self):
        return self.fit

    def save(self, epoch):
        mstate = {
            "models": [m.module if isinstance(m, nn.DataParallel) for m in self.model_list else m],
            "optimizers": self.optimizers,
            "epochs": epoch,
        }
        is_noisy = "_noisy" if self.mixin_noise else ""
        sname = "output/states/{}{}_{}.pt".format(self.model_name, is_noisy, epoch+1)
        torch.save(mstate, sname)

    def precompute(self, m):
        if "resnet" in self.model_name:
            dl = data.DataLoader(self.ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers, shuffle=False)
            m.eval()
            for splt in ["train", "valid"]:
                self.ds.set_split(splt)
                c = self.ds.splits[splt].start
                for i, (mb, tgts) in enumerate(dl):
                    bs = mb.size(0)
                    if self.use_cuda:
                        mb = mb.cuda()
                    mb = Variable(mb)
                    m.zero_grad()
                    out = m(mb).data.cpu()
                    for j_i, j_k in enumerate(range(c, c+bs)):
                        idx_split = self.ds.splits[splt][j_k]
                        k = self.ds.data[idx_split]
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
        labels = torch.LongTensor(labels).unsqueeze(0)
    return (sigs, lengths), labels

def pad_sig(s, max_len, n_feats):
    s_len = s.size(0)
    s_new = s.new(max_len, n_feats).fill_(0)
    s_new[:s_len] = s
    return s_new
