import random
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from collections import Counter

# this function loads the saved embeddings for further use
def load_embeddings(filename):
    genes, embeddings = [],[]
    for line in open(filename, "r").readlines()[1:]: # skipping first line with dimension info
        splits = line.split()
        genes.append(splits[0].strip())
        embeddings.append(np.asarray(splits[1:], dtype = "float32"))
    return np.asarray(genes), np.asarray(embeddings)

# this function checks if the given directory exists and creates one if not present
def check_dir(dirpath):
    return os.path.exists(dirpath)

def load_tm_pairs(dirpath):
    #dirpath = '/var/www/ghai7c/IPF/other_gene2vec_inputs/'
    tm_pairs = np.load(os.path.join(dirpath, 'tabula_muris_gene_pairs.npy'))
    labels = np.load(os.path.join(dirpath, 'tabula_muris_gene_labels.npy'))
    return tm_pairs, labels

def filter_tm_pairs(gene2idmap, tm_gene_pairs, tm_labels):
    filtered_tm_pairs, filtered_tm_labels = [],[]
    for idx, pair in enumerate(tm_gene_pairs):
        if pair[0] in gene2idmap and pair[1] in gene2idmap:
            filtered_tm_pairs.append((gene2idmap[pair[0]], gene2idmap[pair[1]]))
            filtered_tm_labels.append(tm_labels[idx])
    num_classes = len(Counter(tm_labels))
    return np.array(filtered_tm_pairs), np.array(filtered_tm_labels), num_classes

def generate_tm_embeddings(embeddings, tm_pairs, device = "cpu"):
    if device == 'cuda':
        embeddings = embeddings.data.cpu().numpy()
    else:
        embeddings = embeddings.data.numpy()
    embeddings = np.array([np.concatenate((embeddings[pair[0]], embeddings[pair[1]]), axis=0) for pair in tm_pairs])
    return torch.from_numpy(embeddings)

# this function reads the gene pairs generated given a file directory path having the text files
def read_gene_pairs(infile):
    #reading input file containing gene pairs
    #infile = '/var/www/ghai7c/IPF/gene2vec_pairs/cases_gene2vec_pairs_0.9.txt'
    gene_pairs = list()
    print("***** Reading gene pairs from input file *****")
    with open(infile,"r") as f:
        for line in f:
            genes = line.strip().split()
            gene_pairs.append(genes)
        f.close()
    print("Number of gene pairs: ",len(gene_pairs))
    return gene_pairs

def topk_precision(output, target, topk = 3):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    # retrieving topk classes predicted
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_topk = correct[:topk].view(-1).float().sum().item()
    topk_acc = (correct_topk/batch_size) * 100
    return topk_acc

def neural_learner(model, embeddings, labels, nfolds = 5, n_epochs = 100, lr = 0.001, device = "cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    skf = StratifiedKFold(n_splits = nfolds)
    if device == 'cuda':
        model = model.to(device)
    precisions, top3_precisions = [], []
    for fold, (train_data, test_data) in enumerate(skf.split(embeddings, labels)):
        print("\t", "*"*5, "Neural Learner (Fold", fold, ")", "*"*5, "\t")
        X_train, y_train = embeddings[train_data], torch.from_numpy(labels[train_data]).long()
        X_test, y_test = embeddings[test_data], torch.from_numpy(labels[test_data]).long()

        if device == "cuda":
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_test = X_test.to(device)
            y_test = y_test.to(device)

        for i in range(n_epochs):
            optimizer.zero_grad()
            preds = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

        # Computing test loss
        with torch.no_grad():
            test_preds = model(X_test)
            top3_precision = topk_precision(test_preds, y_test, topk = 3)
            precision = topk_precision(test_preds, y_test, topk = 1)
            precisions.append(precision)
            top3_precisions.append(top3_precision)
            print("Fold: %d, Precision %0.8f, Precision@3 %0.8f" %(fold+1, round(precision, 3), round(top3_precision, 3)))
        # if i%100 == 0:
    return round(np.mean(np.array(precisions)), 3), round(np.mean(np.array(top3_precisions)), 3)