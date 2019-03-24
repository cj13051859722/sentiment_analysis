# 中文情感分析

本项目旨在通过一个中文情感分析问题熟悉各种机器学习算法（逻辑回归，支持向量机，朴素贝叶斯等）以及简单的深度学习文本分类方法（BiLSTM、CNN）。

## 数据集

数据集使用的是从大众点评上抓取的用户评价，它的[训练集](https://drive.google.com/file/d/0Bz8a_Dbh9QhbV3l5LW54TWRITDg/view)有两百万条评论及其相应的标注（正面或者负面，这是个二分类问题），[测试集](https://drive.google.com/file/d/0Bz8a_Dbh9QhbYVdVcnZVXzRmUkU/view)有五十万条数据。这里使用的是它的一个子集：60000条训练数据，3000条测试数据。数据文件就放在`datasets`目录下面。

## 结果

下面是各种不同的分类方法在该数据集上的结果（作为对比，当前[最好的模型](https://arxiv.org/abs/1901.10125)的准确率大约为78.46%)：

|      | 逻辑回归   | 支持向量机  | 朴素贝叶斯  | 随机森林   | CNN    | Bi-LSTM |
| ---- | ------ | ------ | ------ | ------ | ------ | ------- |
| 准确率  | 74.43% | 74.47% | 70.70% | 67.20% | 66.37% | 66.52%  |

其中深度学习的方法运行效果还不如机器学习算法，可能主要是因为所用的数据比较少，未能使深度模型发挥它的优势（[论文Glyce: Glyph-vectors for Chinese Character Representations](https://arxiv.org/abs/1901.10125) 使用预训练的词向量 +LSTM模型在整个数据集上运行可以达到78.46%的准确率）。

## 运行

1.安装依赖：

 ```shell
pip3 install -r requirement.txt
 ```

2.在数据集上运行机器学习分类算法：

```shell
python3 sklearn_main.py
```

3.在数据集上运行深度学习算法：

```shell
python3 make_vocab.py  # 构建字典，运行之后会在Resources目录下生成vocab.csv文件
python3 main.py # 训练评估CNN、LSTM模型
```

其中CNN以及Bi-LSTM模型的相关参数，以及训练使用的参数可以在`./models/config.py`中配置。

如果要使用预训练的词向量，可以从[这个链接](https://github.com/Embedding/Chinese-Word-Vectors)选择一个预训练的词向量下载，下载之后将它放到`Resources`目录下，并将其命名为`pretrained_embeddings`即可。（查看`utils.py`的`load_embeddings`函数）

## 代码结构说明

```shell
.
├── data.py                # DianPingDataSet类，用于加载数据
├── datasets               # 数据集目录
│   ├── test.csv
│   ├── train.csv
│   └── vocab.csv          # 词典
├── main.py                # 运行CNN,LSTM模型
├── make_vocab.py          # 入训练数据构建词典
├── models                 # 模型的具体实现(目前包含 CNN/LSTM/LR)
│   ├── cnn_clf.py      
│   ├── config.py　　　　　　# 模型参数设置文件
│   ├── deep.py
│   ├── __init__.py
│   ├── logistic_regression.py
│   ├── lstm_clf.py
├── README.md
├── requirement.txt        # 项目所用第三方库
├── Resources
│   └── stopwords.txt      # 禁用词
├── sklearn_main.py        # 调用sklearn包中的机器学习算法
├── utils.py               # 包含一些工具函数
└── voc.py                 # 词典类
```

## TODO

* 实现其他的机器学习算法，而不是调用sklearn。
* 实现更高级的模型:
  * LSTM+Attention
  * [Self Attention](https://github.com/prakashpandey9/Text-Classification-Pytorch)
  * [RCNN](https://github.com/prakashpandey9/Text-Classification-Pytorch)
  * [HAN](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
  * ...
* 在大的数据集上跑一下效果。