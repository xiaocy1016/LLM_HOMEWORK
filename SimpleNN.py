import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.autograd import Variable


class DataloaderClass(Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.len = len(train_x)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.len


batch_size = 1

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super(SentimentLSTM, self).__init__()

        # 定义LSTM层
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # 定义一个dropout层
        self.dropout = nn.Dropout(dropout)

        # 定义全连接层
        # 如果是双向LSTM，hidden_dim需要乘以2
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM层的输出形状为 (batch_size, seq_length, hidden_dim * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)

        # 如果是双向LSTM，我们需要连接最后一个时间步的正向和反向隐藏状态
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # 应用dropout
        out = self.dropout(hidden)

        # 通过全连接层
        out = self.fc(out)

        # 应用sigmoid激活函数
        out = self.sigmoid(out)

        return out


# 定义超参数
embedding_dim = 200  # Word2Vec词向量的维度
hidden_dim = 256  # LSTM隐藏层的维度
output_dim = 1  # 输出维度，1表示二分类问题
n_layers = 2  # LSTM层的数量
bidirectional = True  # 是否使用双向LSTM
dropout = 0.5  # dropout层的比例

# 实例化模型
model = SentimentLSTM(embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

# 打印模型结构
print(model)


def train(dataloader, model, loss_fn, optimizer): #模型训练过程的定义；这个可以看作是模板，以后写 pytorch 程序基本都这样
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = torch.stack(X)
        X = X.to(device)
        y = y.to(device)
        # print(batch)
        # print(X)
        # print(len(X))
        # print(type(X))
        # for t in X:
        #     print(t)
        #     print(type(t))
        # X = [t.to(device) for t in X]
        # print(y)
        # print(type(y))
        # for t in y:
        #     print(t)
        #     print(type(t))
        # y = [y.to(device) for y in y]

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn): #模型测试过程的定义，这个也是模板，以后可以借鉴
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 预处理函数：去除标点符号，转换为小写，分词
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return word_tokenize(text)


def TrainWord2Vec(movie_reviews):
    from gensim.models import Word2Vec

    # 预处理评论数据
    common_texts = [preprocess(review) for review in movie_reviews]
    # print(common_texts)
    # 训练 Word2Vec 模型
    model = Word2Vec(sentences=common_texts, vector_size=10, window=5, min_count=1, workers=4)
    model.save('word2vec.model')
    return model


def TransWord2Vec(model, movie_reviews):
    reviews_vec = []
    for review in movie_reviews:
        review_vec = []
        for word in preprocess(review):
            if word in model.wv.index_to_key:
                review_vec.extend(model.wv[word])
        review_vec.extend([0] * 200)
        reviews_vec.append(review_vec[:200])
    return reviews_vec


def load_data(dataset, labels):
    loader_obj = DataloaderClass(dataset, labels)
    loader = DataLoader(loader_obj, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == '__main__':
    nltk.download('punkt')

    # IMDB 5w条电影评论数据
    datasets = pandas.read_csv('data/IMDB Dataset.csv')
    # print(datasets)
    # print(type(datasets))

    model = Word2Vec.load("word2vec.model")
    # 训练Word2Vec模型
    # model = TrainWord2Vec(datasets['review'].tolist())
    print("模型加载完成")

    # 获取训练数据
    labels = datasets['sentiment'].tolist()[:100]
    final_labels = []
    for label in labels:
        if label == "positive":
            final_labels.append(0)
        elif label == "negative":
            final_labels.append(1)
    final_dataset = TransWord2Vec(model, datasets['review'].tolist()[:100])
    print("数据转换为向量完成")

    train_dataset, test_dataset, labels_train, labels_test = \
        train_test_split(final_dataset, final_labels, stratify=final_labels, test_size=0.2)
    train_dataset, valid_dataset, labels_train, labels_valid = \
        train_test_split(train_dataset, labels_train, stratify=labels_train, test_size=0.125)
    # print(train_dataset)
    for i in train_dataset:
        print(len(i))
    print(train_dataset)
    train_loader = load_data(train_dataset, labels_train)
    valid_loader = load_data(valid_dataset, labels_valid)
    test_loader = load_data(test_dataset, labels_test)
    print("数据加载完成")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = SentimentLSTM(embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()  # 课上我们说过，loss 类型是可以选择的
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 这里的优化器也是可以选择的

    epochs = 5  # 这个训练的轮数也可以设置

    print("开始训练")
    # 下面这个训练和测试的过程也是标准形式，我们用自己的数据也还是这样去写
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(valid_loader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model/haha")  # 模型可以保存下来，这里 model 文件夹要和当前 py 文件在同一个目录下
    print("Saved PyTorch Model State to the project root folder!")

    # classes = [
    #     "T-shirt/top",
    #     "Trouser",
    #     "Pullover",
    #     "Dress",
    #     "Coat",
    #     "Sandal",
    #     "Shirt",
    #     "Sneaker",
    #     "Bag",
    #     "Ankle boot",
    # ]
    # model.eval()
    # x, y = test_data[0][0], test_data[0][1]
    # with torch.no_grad():
    #     x = x.to(device)
    #     pred = model(x)
    #     predicted, actual = classes[pred[0].argmax(0)], classes[y]
    #     print(f'Predicted: "{predicted}", Actual: "{actual}"')