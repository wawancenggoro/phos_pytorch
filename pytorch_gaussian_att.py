import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import math

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import torch

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# tf.enable_eager_execution()

# Read sample from Dataset

with open('dataset/PELM/fixed_sequences_length_9/Group_Phos_S_pos.fasta', 'r') as f:
    PELM_s_positif_txt = f.readlines()
with open('dataset/PELM/fixed_sequences_length_9/Group_Phos_T_pos.fasta', 'r') as f:
    PELM_t_positif_txt = f.readlines()
with open('dataset/PELM/fixed_sequences_length_9/Group_Phos_Y_pos.fasta', 'r') as f:
    PELM_y_positif_txt = f.readlines()
with open('dataset/PPA/fixed_sequences_length_9/S_IDS_pos.fasta', 'r') as f:
    PPA_s_positif_txt = f.readlines()
with open('dataset/PPA/fixed_sequences_length_9/T_IDS_pos.fasta', 'r') as f:
    PPA_t_positif_txt = f.readlines()
with open('dataset/PPA/fixed_sequences_length_9/Y_IDS_pos.fasta', 'r') as f:
    PPA_y_positif_txt = f.readlines()

with open('dataset/PELM/fixed_sequences_length_9/Group_Phos_S_neg.fasta', 'r') as f:
    PELM_s_negatif_txt = f.readlines()
with open('dataset/PELM/fixed_sequences_length_9/Group_Phos_T_neg.fasta', 'r') as f:
    PELM_t_negatif_txt = f.readlines()
with open('dataset/PELM/fixed_sequences_length_9/Group_Phos_Y_neg.fasta', 'r') as f:
    PELM_y_negatif_txt = f.readlines()
with open('dataset/PPA/fixed_sequences_length_9/S_IDS_neg.fasta', 'r') as f:
    PPA_s_negatif_txt = f.readlines()
with open('dataset/PPA/fixed_sequences_length_9/T_IDS_neg.fasta', 'r') as f:
    PPA_t_negatif_txt = f.readlines()
with open('dataset/PPA/fixed_sequences_length_9/Y_IDS_neg.fasta', 'r') as f:
    PPA_y_negatif_txt = f.readlines()

# Pick the window 9

PELM_s_positif = np.array([])
for i in range(1,len(PELM_s_positif_txt),2):
    temp = PELM_s_positif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PELM_s_positif = np.append(PELM_s_positif, temp2)
print('PELM Dataset, S positive shape: ', PELM_s_positif.reshape(int(len(PELM_s_positif)/9),9).shape)

PELM_t_positif = np.array([])
for i in range(1,len(PELM_t_positif_txt),2):
    temp = PELM_t_positif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PELM_t_positif = np.append(PELM_t_positif, temp2)
print('PELM Dataset, T positive shape: ', PELM_t_positif.reshape(int(len(PELM_t_positif)/9),9).shape)
    
PELM_y_positif = np.array([])
for i in range(1,len(PELM_y_positif_txt),2):
    temp = PELM_y_positif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PELM_y_positif = np.append(PELM_y_positif, temp2)
print('PELM Dataset, Y positive shape: ', PELM_y_positif.reshape(int(len(PELM_y_positif)/9),9).shape)

PPA_s_positif = np.array([])
for i in range(1,len(PPA_s_positif_txt),2):
    temp = PPA_s_positif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PPA_s_positif = np.append(PPA_s_positif, temp2)
print('PPA Dataset, S positive shape: ', PPA_s_positif.reshape(int(len(PPA_s_positif)/9),9).shape)

PPA_t_positif = np.array([])
for i in range(1,len(PPA_t_positif_txt),2):
    temp = PPA_t_positif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PPA_t_positif = np.append(PPA_t_positif, temp2)
print('PPA Dataset, T positive shape: ', PPA_t_positif.reshape(int(len(PPA_t_positif)/9),9).shape)
    
PPA_y_positif = np.array([])
for i in range(1,len(PPA_y_positif_txt),2):
    temp = PPA_y_positif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PPA_y_positif = np.append(PPA_y_positif, temp2)
print('PPA Dataset, Y positive shape: ', PPA_y_positif.reshape(int(len(PPA_y_positif)/9),9).shape)

print()

PELM_s_negatif = np.array([])
for i in range(1,len(PELM_s_negatif_txt),2):
    temp = PELM_s_negatif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PELM_s_negatif = np.append(PELM_s_negatif, temp2)
print('PELM Dataset, S negative shape: ', PELM_s_negatif.reshape(int(len(PELM_s_negatif)/9),9).shape)

PELM_t_negatif = np.array([])
for i in range(1,len(PELM_t_negatif_txt),2):
    temp = PELM_t_negatif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PELM_t_negatif = np.append(PELM_t_negatif, temp2)
print('PELM Dataset, T negative shape: ', PELM_t_negatif.reshape(int(len(PELM_t_negatif)/9),9).shape)
    
PELM_y_negatif = np.array([])
for i in range(1,len(PELM_y_negatif_txt),2):
    temp = PELM_y_negatif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PELM_y_negatif = np.append(PELM_y_negatif, temp2)
print('PELM Dataset, Y negative shape: ', PELM_y_negatif.reshape(int(len(PELM_y_negatif)/9),9).shape)

PPA_s_negatif = np.array([])
for i in range(1,len(PPA_s_negatif_txt),2):
    temp = PPA_s_negatif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PPA_s_negatif = np.append(PPA_s_negatif, temp2)
print('PPA Dataset, S negative shape: ', PPA_s_negatif.reshape(int(len(PPA_s_negatif)/9),9).shape)

PPA_t_negatif = np.array([])
for i in range(1,len(PPA_t_negatif_txt),2):
    temp = PPA_t_negatif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PPA_t_negatif = np.append(PPA_t_negatif, temp2)
print('PPA Dataset, T negative shape: ', PPA_t_negatif.reshape(int(len(PPA_t_negatif)/9),9).shape)
    
PPA_y_negatif = np.array([])
for i in range(1,len(PPA_y_negatif_txt),2):
    temp = PPA_y_negatif_txt[i]
    temp1 = temp[0:9]
    temp2 = list(temp1)
    PPA_y_negatif = np.append(PPA_y_negatif, temp2)
print('PPA Dataset, Y negative shape: ', PPA_y_negatif.reshape(int(len(PPA_y_negatif)/9),9).shape)

# Choose Dataset to train, make sure correspond with negative dataset

dataset_pos = PELM_s_positif
dataset_neg = PELM_s_negatif
string_name = 'PELM_s'

# dataset_pos = np.hstack((PELM_s_positif, PELM_t_positif, PELM_y_positif,
#     PPA_s_positif, PPA_t_positif, PPA_y_positif))
# dataset_neg = np.hstack((PELM_s_negatif, PELM_t_negatif, PELM_y_negatif,
#     PPA_s_negatif, PPA_t_negatif, PPA_y_negatif))
# string_name = 'All'

# Expand dimension, Reshape and Create Label

sequenceLP = int(len(dataset_pos)/9)
dataset_pos = np.expand_dims(dataset_pos, axis=0)
dataset_pos = dataset_pos.reshape(sequenceLP,9)
label_pos = np.ones((sequenceLP,), dtype=int)
label_pos = np.expand_dims(label_pos, axis=0)
label_pos = label_pos.reshape(sequenceLP,1)

sequenceLN = int(len(dataset_neg)/9)
dataset_neg = np.expand_dims(dataset_neg, axis=0)
dataset_neg = dataset_neg.reshape(sequenceLN,9)
label_neg = np.zeros((sequenceLN,), dtype=int)
label_neg = np.expand_dims(label_neg, axis=0)
label_neg = label_neg.reshape(sequenceLN,1)

# Validate

print('Positive Dataset shape: ', dataset_pos.shape)
print('Positive Label shape: ', label_pos.shape)
print('Negative Dataset shape: ', dataset_neg.shape)
print('Negative Label shape: ', label_neg.shape)

# Dataset preparation

dataset_X = np.concatenate((dataset_pos, dataset_neg), axis=0, out=None)
dataset_Y = np.concatenate((label_pos, label_neg), axis=0, out=None)

# Tokenizing, Unique character got its own number

asam = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(asam)
dataset_X_token = []
for i in range(len(dataset_X)):
    temp = tokenizer.texts_to_sequences(dataset_X[i])
    dataset_X_token = np.append(dataset_X_token, temp)

dataset_X_token = dataset_X_token-1
dataset_X_token = dataset_X_token.reshape(len(dataset_X),9)

# Onehot

dataset_X_token_onehot = to_categorical(dataset_X_token)
dataset_X_token_onehot = np.expand_dims(dataset_X_token_onehot, axis=3)
dataset_X_token_onehot = dataset_X_token_onehot.reshape(len(dataset_X),9,20,1)

# dataset_Y_onehot = to_categorical(dataset_Y)
dataset_Y_onehot = dataset_Y

# Shuffle Dataset, devide

main_X, main_Y = shuffle(dataset_X_token, dataset_Y_onehot, random_state=13)
train_X, valid_X, train_Y, valid_Y = train_test_split(dataset_X_token, dataset_Y_onehot, 
                                                              test_size=0.2, random_state=13)

# Validation

print('main X shape: ', main_X.shape)
print('main Y shape: ', main_Y.shape)
print('train X shape: ', train_X.shape)
print('train Y shape: ', train_Y.shape)
print('valid X shape: ', valid_X.shape)
print('valid Y shape: ', valid_Y.shape)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

seq_pos = np.linspace(0,8,9)
mu = 4
sig = 4
gaussian(seq_pos, mu, sig)

class Net(torch.nn.Module):
  def __init__(self, droprate):
    super(Net, self).__init__()
    self.embed = torch.nn.Embedding(20, 16)

    self.att_fc = torch.nn.Linear(9*16, 9)
    self.softmax = torch.nn.functional.softmax

    self.fc1 = torch.nn.Linear(9*16, 128)
    self.fc2 = torch.nn.Linear(128, 128)
    self.relu = torch.nn.functional.relu

    self.last_fc = torch.nn.Linear(128, 2)

    self.droprate = droprate
    if self.droprate>0:
        self.dropout = torch.nn.Dropout(droprate) 

  def forward(self, x):
    emb = self.embed(x)

    # simple attention
    out = emb.view((x.shape[0], -1))
    # att = self.softmax(self.att_fc(out), dim=1)

    # # Gaussian hard code
    # att = att = torch.FloatTensor(gaussian(seq_pos, mu, sig)).unsqueeze(0).repeat(x.shape[0],1)
    
    # out = torch.mul(emb, att.unsqueeze(2)).view((x.shape[0], -1))
    out = self.relu(self.fc1(out))
    if self.droprate>0:
        out = self.dropout(out)
    out = self.relu(self.fc2(out))
    if self.droprate>0:
        out = self.dropout(out)
    out = self.last_fc(out)
    # out = torch.nn.functional.softmax(out, dim=1)
    return out

batch_size = 32
droprate = 0

model = Net(droprate=droprate).cuda()
train_X = torch.LongTensor(train_X).cuda()
train_Y = torch.squeeze(torch.LongTensor(train_Y),1).cuda()
valid_X = torch.LongTensor(valid_X).cuda()
valid_Y = torch.squeeze(torch.LongTensor(valid_Y),1).cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# import IPython; IPython.embed()

best_val_loss = float("inf")
for i in range(10):
    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_Y = train_Y[perm]
    for j in range(train_X.shape[0]//batch_size):
        out = model(train_X[j*batch_size:(j+1)*batch_size])
        loss = loss_fn(out, train_Y[j*batch_size:(j+1)*batch_size])
        model.zero_grad()
        loss.backward()
        optimizer.step()

    _, pred = torch.nn.functional.softmax(out, dim=1).max(1)

    train_loss = loss_fn(model(train_X), train_Y).cpu().data.tolist()
    val_loss = loss_fn(model(valid_X), valid_Y).cpu().data.tolist()
    print((train_loss, val_loss))

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth.tar')
        best_val_loss = val_loss

model.load_state_dict(torch.load('best_model.pth.tar'))
out = model(valid_X)
_, pred = torch.nn.functional.softmax(out, dim=1).max(1)
y_pred = pred.cpu().numpy()
y_true = valid_Y.cpu().numpy()

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
f1 = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
auc = roc_auc_score(y_true, y_pred, average='macro', sample_weight=None, max_fpr=None)
sensi = tp/(tp+fn)
specificity = tn/(tn+fp)
accu = (tn + tp)/(tn + tp + fn + fp)
mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
print('{} Result'.format(string_name))
print('Accuracy :', accu)
print('AUC :', auc)
print('Sensitivity :', sensi)
print('Specificity :', specificity)
print('F1 :', f1)
print('MCC :', mcc)

import IPython; IPython.embed()