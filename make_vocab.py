from collections import Counter

from data import DianPingDataSet
from voc import Voc

VOC_FILE = "./datasets/vocab.csv"


def make_vocab(path):
    voc = Voc()
    train_dataset = DianPingDataSet("train")
    for _, sentence in train_dataset:
        voc.add_sentence(sentence)

    counter = Counter(voc.gram2count)
    with open(path, "w") as f:
        for word, count in counter.most_common():
            f.write(word + ',' + str(count) + '\n')
    print("Build Vocab Done!")


if __name__ == "__main__":
    make_vocab(VOC_FILE)
