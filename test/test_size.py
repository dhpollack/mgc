import unittest
import torchaudio.transforms as transforms
import mgc_transforms
from loader_audioset import *

class Test_SIZE(unittest.TestCase):
    bdir = "data/audioset"

    def test1(self):
        """
            Test that the data loader does transforms
        """

        NMELS = 224


        ds = AUDIOSET(self.bdir, randomize=True)
        T = transforms.Compose([transforms.PadTrim(ds.maxlen),
                                mgc_transforms.MEL(n_mels=NMELS),
                                mgc_transforms.BLC2CBL()])
        TT = mgc_transforms.BinENC(ds.labels_dict)
        ds.transform = T
        ds.target_transform = TT
        dl = data.DataLoader(ds, collate_fn=bce_collate, batch_size = 5)
        labels_total = 0
        for i, (a, b) in enumerate(dl):
            print(a.size(), b.size())
            break


        self.assertTrue(a.size()[-2:] == (NMELS, 313))

if __name__ == '__main__':
    unittest.main()
