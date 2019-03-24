from os.path import join

from torch.utils.data import Dataset


class DianPingDataSet(Dataset):
    def __init__(self, split, data_dir="./datasets/"):
        assert split in ["train", "test"]
        self.split = split
        self.data_dir = data_dir

        self.pairs = self.load_data()

    def load_data(self):
        pairs = []
        with open(join(self.data_dir, self.split+".csv")) as f:
            for line in f:
                label, sentence = line.split(",", 1)
                label = label.strip('"')
                sentence = sentence.strip('"').strip("\n")

                label = int(label) - 1  # 将1和2的label转化到0和1
                pairs.append((label, sentence))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]
