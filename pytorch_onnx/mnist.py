from pathlib import Path

import pickle
import gzip
import requests
import torch
import math

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    CONTENT = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(CONTENT)

with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
    ((x_train,y_train), (x_valid, y_valid),_) = pickle.load(f,encoding="latin-1")


x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape

weights = torch.randn(784,10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10,requires_grad=True)

def log_softmax(x):
    return x-x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x

print(xb.shape, xb.exp())
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)

def nll(input, target):
    print(input.shape, target.shape)
    print(input[range(target.shape[0]), target])
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


yb = y_train[0:bs]
print(loss_func(preds, yb))




