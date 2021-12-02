#!/usr/bin/python
#-- coding:utf8 --
import sys
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
import gzip
import pickle
import timeit
import argparse

if torch.cuda.is_available():
        cuda = True
        #torch.cuda.set_device(1)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')
#cuda = False        
def padding_sequence_new(seq, window_size = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < window_size:
        gap_len = window_size -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

def read_rna_dict(rna_dict = 'rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def split_overlap_seq(seq, window_size):
    overlap_size = 50
    #pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels

def get_RNA_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def get_bag_data(data, channel = 7, window_size = 101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        #flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        if num_of_ins > channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                #bag_subt.append(random.choice(bag_subt))
                tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))

    return bags, labels


def get_bag_data_1_channel(data, max_len = 501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        #pdb.set_trace()
        #bag_seqs = split_overlap_seq(seq)
        bag_seq = padding_sequence(seq, max_len = max_len)
        #flat_array = []
        bag_subt = []
        #for bag_seq in bag_seqs:
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        # print tri_fea
        bag_subt.append(tri_fea.T)
        # print tri_fea.T
        
        bags.append(np.array(bag_subt))
        # print bags
        
    return bags, labels

def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []
        for idx, (X, y) in enumerate(train_loader):
            #for X, y in zip(X_train, y_train):
            #X_v = Variable(torch.from_numpy(X.astype(np.float32)))
             #y_v = Variable(torch.from_numpy(np.array(ys)).long())
            X_v = Variable(X)
            y_v = Variable(y)
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
                
            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item()) # need change to loss_list.append(loss.item()) for pytorch v0.4 or above

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        #X_list = batch(X, batch_size)
        #y_list = batch(y, batch_size)
        #pdb.set_trace()
        print (X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                              torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            # print ('%.5f'%loss)
            #rint("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

    def evaluate(self, X, y, batch_size=32):
        
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)
        #lasses = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        #cc = self._accuracy(classes, y)
        return loss.data[0], auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X= X.cuda()        
        y_pred = self.model(X)
        return y_pred        

    def predict_proba(self, X):
        self.model.eval()
        return self.model.predict_proba(X)
        
class CNN(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        out1_size = int((window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool_size = int((out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size = (1, 10), stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride = stride))
        out2_size = int((maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        self.drop1 = nn.Dropout(p=0.25)
        print ('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(maxpool2_size*nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
    
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]


def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        # print(labels)
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True):
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)
    
    return train_bags, label


def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    else:
        print ('only support CNN model')

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)
    torch.save(model.state_dict(), model_file)
    #print 'predicting'         
    #pred = model.predict_proba(test_bags)
    #return model

def predict_network(model_type, X_test, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    else:
        print ('only support CNN model')

    if cuda:
        model = model.cuda()
                
    model.load_state_dict(torch.load(model_file))
    try:
        pred = model.predict_proba(X_test)
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred
        
def run_ideepe(parser):
    #data_dir = './GraphProt_CLIP_sequences/'
    posi = parser.posi
    nega = parser.nega
    model_type = parser.model_type
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    start_time = timeit.default_timer()
    #pdb.set_trace() 
    if predict:
        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True')
            return

    if train:
        
        print("101")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 7, window_size = 101)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_file + '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        print("151")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 4, window_size = 151)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 4, window_size = 151 + 6, model_file = model_file + '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
            
        print("201")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 3, window_size = 201)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_file + '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        print("251")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 251)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 251 + 6, model_file = model_file + '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
            
        print("301")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 301)
        # print(np.array(train_bags).shape)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_file + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        
        print("351")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 351)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 351 + 6, model_file = model_file + '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
            
        print("401")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 401)
        # print(np.array(train_bags).shape)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_file + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        
        print("451")
        train_bags, train_labels = [], []
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 451)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 451 + 6, model_file = model_file + '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
            
        print ('501')
        file_out = open('time_train.txt','a')
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 501)
        model = train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_file + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        end_time = timeit.default_timer()
        file_out.write(str(round(float(end_time - start_time),3))+'\n')
        file_out.close()
        # print ("Training final took: %.2f min" % float((end_time - start_time)/60))
    elif predict:
        fw = open(out_file, 'w')
        file_out = open('pre_auc.txt','a')
        file_out2 = open('time_test.txt', 'a')

        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 501)
        predict1 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 501 + 6, model_file = model_file+ '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
            
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101) 
        predict2 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_file+ '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)       
            
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401) 
        predict3 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_file+ '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)       
            
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301) 
        predict4 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_file+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)       
            
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201) 
        predict5 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_file+ '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)   

        X_test, X_labels = get_data(testfile, nega , channel = 4, window_size = 151) 
        predict6 = predict_network(model_type, np.array(X_test), channel = 4, window_size = 151 + 6, model_file = model_file+ '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters) 

        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 251) 
        predict7 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 251 + 6, model_file = model_file+ '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters) 

        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 351) 
        predict8 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 351 + 6, model_file = model_file+ '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters) 

        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 451) 
        predict9 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 451 + 6, model_file = model_file+ '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)     
            
        predict = (predict1 + predict2 + predict3 + predict4 + predict5 + predict6 + predict7 + predict8 + predict9)/9.0
        # print predict
	    # pdb.set_trace()
        auc = roc_auc_score(X_labels, predict)
        print ('AUC:{:.3f}'.format(auc))        
        myprob = "\n".join(map(str, predict))  
        fw.write(myprob)
        fw.close()
        file_out.write(str(round(float(auc),3))+'\n')
        file_out.close()
        end_time = timeit.default_timer()
        file_out2.write(str(round(float(end_time - start_time),3))+'\n')
        file_out2.close()
    else:
        print ('please specify that you want to train the mdoel or predict for your own sequences')


def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>', help='The fasta file of negative training samples')
    parser.add_argument('--model_type', type=str, default='CNN', help='The default model is CNN')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--train', type=bool, default=True, help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl', help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',  help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--num_filters', type=int, default=16, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')

    args = parser.parse_args()    #解析添加的参数
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print (args)
#model_type = sys.argv[1]
run_ideepe(args)
# run_ideepe_on_graphprot()


