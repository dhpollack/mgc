import unittest
import torchaudio.transforms as transforms
import mgc_transforms
from loader_audioset import *

class Test_AUDIOSET(unittest.TestCase):
    bdir = "data/audioset"

    def test1(self):
        ds = AUDIOSET(self.bdir, randomize=False)
        for i, (a, b) in enumerate(ds):
            print(a.size(), b)
            if i > 10: break

    def test2(self):
        """
            Test the max_len function
        """
        ds = AUDIOSET(self.bdir)
        maxlen = 0
        for i, (a, b) in enumerate(ds):
            print(b)
            maxlen = a.size(0) if a.size(0) > maxlen else maxlen
        print(maxlen)
        ds.find_max_len()
        self.assertEqual(maxlen, ds.maxlen)

    def test3(self):
        """
            Test that the data loader does transforms
        """
        ds = AUDIOSET(self.bdir, randomize=True)
        T = transforms.Compose([transforms.PadTrim(ds.maxlen),
                                mgc_transforms.MEL(),
                                mgc_transforms.BLC2CBL()])
        TT = mgc_transforms.BinENC(ds.labels_dict)
        ds.transform = T
        ds.target_transform = TT
        dl = data.DataLoader(ds, collate_fn=bce_collate, batch_size = 5)
        labels_total = 0
        print(ds.labels_dict)
        for i, (a, b) in enumerate(dl):
            print(a.size(), b.size())
            if i > 10: break

    def test4(self):
        ds = AUDIOSET(self.bdir)
        T = transforms.Compose([transforms.PadTrim(ds.maxlen),])
        TT = mgc_transforms
        vx.transform = T
        dl = data.DataLoader(vx, collate_fn=bce_collate, batch_size = 5)
        total_train = 0
        for i, (mb, l) in enumerate(dl):
            total_train += l.size(0)
            if i == 2:
                #ds.set_split("valid")
                total_valid = 0
                for mb_valid, l_valid in dl:
                    total_valid += l_valid.size(0)
                print(total_valid)
        print(total_train)

if __name__ == '__main__':
    unittest.main()
