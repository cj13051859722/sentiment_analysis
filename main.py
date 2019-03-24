from functools import partial

from torch.utils.data import DataLoader

from data import DianPingDataSet
from utils import collate_fn_ml, collate_fn_dl, load_word2id, load_embeddings
from models.logistic_regression import LogisticRegression
from models.deep import DeepModel

VOCAB_SIZE = 3500  # 指定字典大小
SENT_MAX_LEN = 128  # 指定句子最长的长度


def main():

    # 在训练集上构建一元和二元词典
    word2id = load_word2id(length=VOCAB_SIZE)

    # 为深度学习算法准备数据loader
    train_loader_dl = DataLoader(
        dataset=DianPingDataSet("train"),
        batch_size=64,
        collate_fn=partial(collate_fn_dl, word2id, SENT_MAX_LEN)
    )
    test_loader_dl = DataLoader(
        dataset=DianPingDataSet("test"),
        batch_size=64,
        collate_fn=partial(collate_fn_dl, word2id, SENT_MAX_LEN)
    )
    vocab_size = len(word2id)
    print("Vocab Size:", vocab_size)
    print("加载词向量....")
    try:
        embedding = load_embeddings(word2id)
    except FileNotFoundError:
        embedding = None

    # 在深度学习模型上训练测试(CNN, LSTM)
    print("在BiLSTM模型上训练...")
    lstm_model = DeepModel(vocab_size, embedding, method="lstm")
    lstm_model.train_and_eval(train_loader_dl, test_loader_dl)

    print("在CNN模型上训练...")
    cnn_model = DeepModel(vocab_size, embedding, method="cnn")
    cnn_model.train_and_eval(train_loader_dl, test_loader_dl)

    # # 为机器学习算法准备数据loader
    # 与sklearn_main.py文件中不一样的是，以下的模型是自己实现的(用于学习），
    # 可以作为对比，看看效果
    # train_loader_ml = DataLoader(
    #     dataset=DianPingDataSet("train"),
    #     batch_size=64,
    #     collate_fn=partial(collate_fn_ml, word2id)
    # )
    # test_loader_ml = DataLoader(
    #     dataset=DianPingDataSet("test"),
    #     batch_size=64,
    #     collate_fn=partial(collate_fn_ml, word2id)
    # )

    # # 在LR模型（自己实现的）上训练 测试，简单起见这里使用词袋模型
    # print("使用LR模型进行分类...")
    # input_size = len(word2id)
    # lr_model = LogisticRegression(input_size)
    # lr_model.train_and_eval(train_loader_ml, test_loader_ml)


if __name__ == "__main__":
    main()
