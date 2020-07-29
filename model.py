from gene_pairs import GenePairs
import numpy as np
from Learner import Learner
import torch
import torch.optim as optim
from tqdm import tqdm
import os

class EmbeddingGenerator:
    def __init__(self, data, outfile, emb_dim = 128, batch_size = 1000, neg_samp_cnt = 5,num_iters = 10,
                 init_lr = 0.05, device = "cpu", unsup_loss = "neg_sampling", unigram_flag = True, sup_flag = True,
                 save_every = 5, pre_emb_weights = []):
        '''
        Parameters:
            data: instance of the GenePairs class
            outfile: Output filename where the gene embeddings have to be saved
            emd_dim: dimension size of the embeddings
            batch_size: num_of_gene_pairs in a batch
            neg_samp_cnt: number of negative samples to be used
            num_iters: number of iterations
            init_lr: initial learning rate
            unsup_loss : unsupervised loss function to be used for the skipgram model; default: negative sampling
            sup_flag : flag indicating whether functional supervised loss has to be used
            pre_emb_weights: pre-trained weights in case of pre-trained learning
        '''
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.neg_sam_cnt = neg_samp_cnt
        self.num_iters = num_iters
        self.init_lr = init_lr
        self.device = device
        self.unsup_loss = unsup_loss # Flag indicating the unsupervised loss function to be used
        self.super_flag = sup_flag
        self.unigram_flag = unigram_flag

        self.data = data
        if self.unigram_flag:
            self.neg_generator = self.data.get_neg_samples
        else:
            self.neg_generator = self.data.get_neg_samples_combined

        self.save_every = save_every

        # initiating parameters associated with the supervised trainer
        if self.super_flag:
            pos_labels = torch.sum(self.data.fun_labels)

            # weight added to the overall supervised loss, due to potential imbalance in the overall functional pair and the co-expressed pair counts
            sup_weight = torch.round(self.data.pair_cnt/self.data.fun_pair_cnt)
            self.sup_weight = sup_weight if sup_weight > 1.0 else torch.tensor(1.0).to(self.device)
            neg_labels = self.data.fun_labels.shape[0] - pos_labels
            self.pos_weights = int(neg_labels / pos_labels)  # https://discuss.pytorch.org/t/about-bcewithlogitslosss-pos-weights/22567/5
        else:
            self.sup_weight = torch.tensor(0)
            self.pos_weights = 0

        self.outfile = outfile

        if len(pre_emb_weights)>0:
            pre_emb_weights = torch.from_numpy(pre_emb_weights)

        self.learner = Learner(self.data.vocab_size, self.emb_dim, super_flag=self.super_flag, unsup_loss = self.unsup_loss,
                                 super_weight = self.sup_weight, pre_embeddings = pre_emb_weights, p_dropout=0.5, device = self.device)
        self.learner = self.learner.to(self.device)
        self.optimizer = optim.SGD(self.learner.parameters(), lr = self.init_lr)

        # learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 100000, gamma = 0.5)

    def train(self):
        '''
        Function which performs the training steps
        '''
        num_batches = int(self.data.pair_cnt.item()/self.batch_size)
        # num_batches = 10
        losses, tm_top3_precisions, tm_precisions = [],[],[]
        for iter in range(self.num_iters):
            print("\n","*"*20, "Iteration:",(iter+1),"*"*20)
            process_bar = tqdm(range(num_batches))
            self.data.reset_batch_index()
            outfile = self.outfile + "_iter_" + str(iter+1) + ".txt"
            iter_loss_total = 0
            for i in process_bar:
                batch_gene_pairs, batch_labels = self.data.fetch_batch_pairs(self.batch_size, self.super_flag)
                batch_neg_pairs = self.neg_generator(batch_gene_pairs, self.neg_sam_cnt)
                self.optimizer.zero_grad()
                batch_losses = self.learner.forward(batch_gene_pairs, batch_neg_pairs, batch_labels, self.pos_weights)
                # backward step on the total loss (unsup + sup)
                batch_losses[0].backward()
                self.optimizer.step()
                self.scheduler.step()
                process_bar.set_description("Epoch: %d, Total Loss: %0.6f"
                                % (i+1, round(batch_losses[0].item(), 3)/self.batch_size,))
                iter_loss_total += round(batch_losses[0].item(), 3)/self.batch_size

            iter_loss_total = iter_loss_total/num_batches
            losses.append(iter_loss_total)
            print("\n", "*" * 20, "Avg. Total Loss:", iter_loss_total, "*" * 20)
            
            if (iter+1)%self.save_every == 0:
                print("\n", "*" * 20, "Saving Embeddings", "*" * 20)
                self.learner.save_embeddings(self.data.id2gene, outfile, self.device)

        return losses