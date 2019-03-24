from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np

from data import DianPingDataSet
from utils import preprocess_for_ml
# 直接调用Sklearn中的包 来比较效果


def main():
    # 处理数据
    train_dataset = DianPingDataSet("train")
    test_dataset = DianPingDataSet("test")
    train_labels, train_sents = zip(*train_dataset.pairs)
    test_labels, test_sents = zip(*test_dataset.pairs)

    # 将句子分词，这样因为sklearn是根据空格来判断词语之间的界限的
    train_sents = preprocess_for_ml(train_sents)
    test_sents = preprocess_for_ml(test_sents)

    # 转换为向量的形式，我们使用词的tf-idf值作为特征
    # 处理中文的时候需要指定token_pattern参数，因为sklearn中默认丢弃长度为1的token
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_sents_tfidf = tfidf.fit_transform(train_sents)
    test_sents_tfidf = tfidf.transform(test_sents)

    # 数据准备好之后，开始进行训练！

    # 先尝试一下逻辑斯蒂回归
    lr_clf = LogisticRegression(solver="lbfgs", max_iter=3000)
    lr_clf.fit(train_sents_tfidf, train_labels)
    predicted = lr_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("Accuracy of LogisticRegression: {:.2f}%".format(acc * 100))

    # 朴素贝叶斯：
    nb_clf = MultinomialNB()
    nb_clf.fit(train_sents_tfidf, train_labels)
    predicted = nb_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("Accuracy of Naive Bayes: {:.2f}%".format(acc * 100))

    # 支持向量机
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
    sgd_clf.fit(train_sents_tfidf, train_labels)
    predicted = sgd_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("Accuracy of SVM: {:.2f}%".format(acc * 100))

    # K近邻
    kn_clf = KNeighborsClassifier()
    kn_clf.fit(train_sents_tfidf, train_labels)
    predicted = kn_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("Accuracy of KNN: {:.2f}%".format(acc * 100))

    # 随机森林
    rf_clf = RandomForestClassifier(n_estimators=20)
    rf_clf.fit(train_sents_tfidf, train_labels)
    predicted = rf_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("Accuracy of RandomForest: {:.2f}%".format(acc * 100))

    # K均值 需要运行很久的时间，并且效果不好
    # km_clf = KMeans(n_clusters=2).fit(train_sents_tfidf)
    # predicted = km_clf.predict(test_sents_tfidf)
    # acc = np.mean(predicted == np.array(test_labels))
    # print("Accuracy of K means: {:.2f}%".format(acc * 100))


if __name__ == "__main__":
    main()
