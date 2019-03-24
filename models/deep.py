import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


from .cnn_clf import CNN_Classifier
from .lstm_clf import LSTM_Classifier
from .config import LSTMTrainingConfig, CNNTrainingConfig

# 控制LSTM以及CNN模型的训练，测试


class DeepModel(object):
    def __init__(self, vocab_size, embedding=None, method="cnn"):
        assert method in ["cnn", "lstm"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.method = method
        if method == "cnn":
            self.model = CNN_Classifier(vocab_size).to(self.device)
            self.epoches = CNNTrainingConfig.epoches
            self.learning_rate = CNNTrainingConfig.learning_rate
            self.print_step = CNNTrainingConfig.print_step
            self.lr_decay = CNNTrainingConfig.factor
            self.patience = CNNTrainingConfig.patience
            self.verbose = CNNTrainingConfig.verbose
        elif method == "lstm":
            self.model = LSTM_Classifier(vocab_size).to(self.device)
            self.epoches = LSTMTrainingConfig.epoches
            self.learning_rate = LSTMTrainingConfig.learning_rate
            self.print_step = LSTMTrainingConfig.print_step
            self.lr_decay = LSTMTrainingConfig.factor
            self.patience = LSTMTrainingConfig.patience
            self.verbose = LSTMTrainingConfig.verbose

        if embedding:  # 如果使用预训练的embedding
            self.model.init_embedding(embedding.to(self.device))

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                              "min",
                                              factor=self.lr_decay,
                                              patience=self.patience,
                                              verbose=self.verbose
                                              )
        self.loss_fn = nn.BCELoss()
        self.best_acc = 0.

    def train_and_eval(self, train_loader, test_loader):
        """训练评估模型"""
        for e in range(1, self.epoches+1):
            print("Epoch {} training...".format(e))
            step = 0
            losses = 0.
            for labels, sentences, lengths in train_loader:
                self.model.train()
                self.optimizer.zero_grad()

                labels = labels.to(self.device)
                sentences = sentences.to(self.device)

                # 计算损失，更新参数
                if self.method == "cnn":
                    probs = torch.sigmoid(self.model(
                        sentences)).squeeze(1)  # [B,]
                elif self.method == "lstm":
                    probs = torch.sigmoid(self.model(
                        sentences, lengths)).squeeze(1)  # [B,]
                loss = self.loss_fn(probs, labels.float())
                losses += loss.item()
                loss.backward()
                self.optimizer.step()

                step += 1
                if step % self.print_step == 0:
                    print("Epoch {}: {}/{} {:.2f}% finished, Loss: {:.4f}".format(
                        e, step, len(train_loader),
                        100 * step/len(train_loader),
                        losses/self.print_step
                    ))
                    losses = 0

            self.test(test_loader)

        print("Best Accuracy: {:.2f}%".format(self.best_acc))

    def test(self, test_loader):
        """计算模型在测试集上的准确率以及损失"""
        count = 0.
        correct_num = 0.
        losses = 0.
        self.model.eval()
        with torch.no_grad():
            for labels, sentences, lengths in test_loader:
                labels = labels.to(self.device)
                sentences = sentences.to(self.device)

                if self.method == "cnn":
                    probs = torch.sigmoid(self.model(
                        sentences)).squeeze(1)  # [B,]
                elif self.method == "lstm":
                    probs = torch.sigmoid(self.model(
                        sentences, lengths)).squeeze(1)  # [B,]
                loss = self.loss_fn(probs, labels.float())
                losses += loss.item()

                pred_labels = torch.round(probs)  # [B, ]
                count += len(labels)
                correct_num += (pred_labels.long() == labels).sum().item()
        acc = correct_num / count
        if acc > self.best_acc:
            self.best_acc = acc
        print("Accuracy: {:.2f}%".format(100*acc))

        avg_loss = losses / len(test_loader)
        self.lr_scheduler.step(avg_loss)
