import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f

class Learner(nn.Module):
    ''''
    emb_dim -> embedding dimension
    vocab_size -> number of genes in the vocabulary
    super_flag -> boolean flag indicating if supervised loss has to be included
    pre_embeddings -> set of pre-trained embedding weights if any
    '''
    def __init__(self, vocab_size, emb_dim, batch_size = 1000, super_flag = True, unsup_loss = False,
                 super_weight = 1, pre_embeddings = [], p_dropout = 0.0, device = "cpu"):
        super(Learner, self).__init__()
        self.supervised = super_flag # flag indicating if supervised loss (based on functional annotations) should be used
        self.unsup_loss = unsup_loss # flag indicating whether adaptive softmax based loss should be used
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device
        self.super_weight = Variable(super_weight, requires_grad = False)
        self.batch_size = batch_size
        self.pre_emb_flag = True if len(pre_embeddings)>0 else False
        self.pre_emb = pre_embeddings
        self.p_dropout = p_dropout
        if p_dropout>0:
            self.drop_out = torch.nn.Dropout(p=self.p_dropout, inplace=False)

        # Embedding weights: vocab_size * embedding_dimension
        self.embeddings_hid = nn.Embedding(vocab_size, emb_dim)

        # Softmax i.e. negative sampling weights: embedding_dimension * vocab_size -> transposed
        self.embeddings_out = nn.Embedding(vocab_size, emb_dim)

        if self.supervised:
            self.sup_weight = super_weight.to(self.device)
            self.pred = nn.Linear(2*self.emb_dim, 1, bias = False)

        self.initialize_weights()
        if self.device == "cuda":
            self.embeddings_hid = self.embeddings_hid.cuda()
            self.embeddings_out = self.embeddings_out.cuda()

    # this function creates the embedding layer weights
    def initialize_weights(self):

        if not self.pre_emb_flag:
            # embedding weights is a uniform distribution in [-0.5/vocab_size, 0.5/vocab_size]
            init_range = 0.5 / self.emb_dim
            self.embeddings_hid.weight.data.uniform_(-init_range, init_range)
        else: # if pre-trained embedding weights used
            self.embeddings_hid.weight.data.copy_(self.pre_emb)

        self.embeddings_out.weight.data.uniform_(-0, 0)
        if self.device == "cuda":
            self.embeddings_hid = self.embeddings_hid.cuda()
            self.embeddings_out = self.embeddings_out.cuda()

    # this function computes loss based on AdaptiveLogSoftmax based on https://arxiv.org/abs/1609.04309
    def adaptive_softmax_loss(self, embs, targets):
        cutoffs = torch.tensor([500, 5000]).to(self.device)  # based on gene frequencies plot
        criterion = nn.AdaptiveLogSoftmaxWithLoss(self.emb_dim, self.vocab_size, cutoffs=cutoffs).to(self.device)
        # converting labels into a 1D tensor
        targets = targets.view(embs.shape[0])
        loss = criterion(nn.Dropout()(embs), targets)
        return loss[1]

    # this function returns the negative loss for a given batch of embeddings
    # https://github.com/Adoni/word2vec_pytorch/blob/master/model.py
    def neg_sampling_loss(self, emb_x, emb_y, emb_neg):

        # dot products of positive pairs
        pos_score = torch.mul(emb_x, emb_y).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = f.logsigmoid(pos_score)  # averaging over the batch

        # batch matrix products (i.e. matrix-matrix products of 3D tensors
        # converting 2D matrix of dimensions batch_size * emb_dim into a 3D tensor of batch_size * emb_dim * 1 (using unsqueeze)
        neg_score = torch.bmm(emb_neg, emb_x.unsqueeze(2)).squeeze()
        neg_score = f.logsigmoid(-1 * neg_score)

        neg_sam_loss = -(torch.sum(pos_score) + torch.sum(neg_score))
        return neg_sam_loss

    # this function computes and returns supervised loss based on functional labels
    # https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452/11
    def supervised_loss(self, emb_x, emb_y, labels, pos_weight):

        emb_combined = torch.cat((emb_x, emb_y), 1) # batch_size * (2*emb_dim)
        logits = self.pred(nn.Dropout()(emb_combined))
        criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        loss = criterion(logits, labels)
        return loss

    # forward step computing loss function/score using negative sampling
    def forward(self, positive_pairs, negative_samples, labels = [], pos_weight = 1.0):
        '''
        Parameters:
            positive_pairs: gene pairs in a batch
            negative_samples: list of negative samples. Number of negative samples = batch_size * negative_sampling_size

        return:
            Loss variable
        '''
        indexes_x = positive_pairs[:,0]
        indexes_y = positive_pairs[:,1]
        indexes_neg = negative_samples

        emb_x = self.embeddings_hid(indexes_x)
        emb_y = self.embeddings_out(indexes_y)
        emb_neg = self.embeddings_out(indexes_neg) if indexes_neg is not None else None
        emb_neg_nce = self.embeddings_out(indexes_neg.flatten()) if indexes_neg is not None else None

        if self.unsup_loss == "neg_sampling":
            unsup_loss = self.neg_sampling_loss(emb_x, emb_y, emb_neg)
        else:
            indexes_y = indexes_y.to(self.device)
            unsup_loss = self.adaptive_softmax_loss(emb_x, indexes_y)

        if self.supervised:
            labels = Variable(labels.reshape(positive_pairs.shape[0],1).float(), requires_grad = False)
            pos_weight = Variable(torch.Tensor([pos_weight]), requires_grad = False)
            labels = labels.to(self.device)
            pos_weight = pos_weight.to(self.device)
            sup_loss = self.super_weight*self.supervised_loss(emb_x, emb_y, labels, pos_weight)
            loss = unsup_loss + sup_loss
            return [loss, unsup_loss, sup_loss]

        return [unsup_loss]

    # saving the embeddings
    def save_embeddings(self, id2gene, outfile, device):
        '''
        Save the gene embeddings to the given file path

        Parameters:
            id2gene: map from gene id to gene symbol
            outfile: file name of the output file
            use_cuda : to distinguish between a GPU or CPU tensor
        '''
        if device == 'cuda':
            embeddings = self.embeddings_hid.weight.cpu().data.numpy()
        else:
            embeddings = self.embeddings_hid.weight.data.numpy()
        fout = open(outfile, "w")
        fout.write('%d %d\n' % (len(id2gene), self.emb_dim))
        for gid, gene in id2gene.items():
            embedding = embeddings[gid]
            embedding = ' '.join(map(lambda x: str(x), embedding))
            fout.write('%s %s\n' % (gene, embedding))