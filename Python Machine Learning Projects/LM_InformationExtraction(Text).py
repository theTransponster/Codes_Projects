# -*- coding: utf-8 -*-
"""Untitled2-2.ipynb

**KEI with SROIE **
"""

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import drive
drive.mount('/content/drive')

!pip install colorama



#Library importation

import glob
import os, sys
import random
from tqdm import tqdm
import pandas as pd
import argparse
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
sys.path.append("/content/drive/My Drive/data")
from my_data import VOCAB, MyDataset, color_print
from my_utils import pred_to_dict, compare_truth

"""*Definition of the train and validate functions by partitioning the data available*"""

dataset = MyDataset(
    dict1="/content/drive/My Drive/data/data_dict4.pth",
    test1="/content/drive/My Drive/data/test_dict.pth",
    val = 76
)

def train(model, dataset, criterion, optimizer, epoch_range, batch_size):
    model.train()

    for epoch in range(*epoch_range):
        optimizer.zero_grad()

        text, truth = dataset.get_train_data(batch_size=batch_size)
        pred = model(text)

        loss = criterion(pred.view(-1, 5), truth.view(-1))
        loss.backward()

        optimizer.step()

        print(f"#{epoch:04d} | Loss: {loss.item():.4f}")

def validate(model, dataset, batch_size=1):
    model.eval()
    with torch.no_grad():
        keys, text, truth = dataset.get_val_data(batch_size=batch_size)

        oupt = model(text)
        prob = torch.nn.functional.softmax(oupt, dim=2)
        prob, pred = torch.max(prob, dim=2)

        prob = prob.cpu().numpy()
        pred = pred.cpu().numpy()

        for i, key in enumerate(keys):
            real_text, _ = dataset.val_dict[key]
            result = pred_to_dict(real_text, pred[:, i], prob[:, i])

            for k, v in result.items():
                print(f"{k:>8}: {v}")

            color_print(real_text, pred[:, i])



#4 layers proposed

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=4, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, 5)



    def forward(self, inpt):
        embedded = self.embed(inpt)
        feature, _ = self.lstm(embedded)
        oupt = self.linear(feature)

        return oupt

#The next set of values were taken from the github model
device = torch.device('cpu')

model = Model(len(VOCAB), 16, 256).to(device)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([0.1, 1, 1.2, 0.8, 1.5], device=device)
)

optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, 1000)

#Train model

for i in range(1500 // 100):
    train(
        model,
        dataset,
        criterion,
        optimizer,
        (i * val_at + 1, (i + 1) * val_at + 1),
        16,
    )

torch.save(model.state_dict(), "model1.pth")

model.load_state_dict(torch.load("Model/model1.pth"))

results_path = "Results2/"

model.eval()
with torch.no_grad():
    for key in dataset.test_dict.keys():
        #counter = counter + 1
        text_tensor = dataset.get_test_data(key)

        oupt = model(text_tensor)
        prob = torch.nn.functional.softmax(oupt, dim=2) #Softmax activation with two units.
        prob, pred = torch.max(prob, dim=2)

        prob = prob.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()

        real_text = dataset.test_dict[key]
        result = pred_to_dict(real_text, pred, prob)

        with open(results_path + key + ".txt", "w", encoding="utf-8") as json_opened:
            json.dump(result, json_opened, indent=4)

        print(key)