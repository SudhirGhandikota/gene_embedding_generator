import numpy as np
from collections import defaultdict
import torch
import torch.cuda as cutorch

class GenePairs:
    '''
    Stores the input gene pairs

    Attributes:
        infile -> filename containing gene pairs
        funinfile -> filename containing functional gene pairs
        min_cnt -> minimum support for a gene (i.e. minimum number of gene pairs a gene has to be in); default: 1
        unigram -> Flag indicating if Unigram distribution has to be constructed; default: True
        pre_trained_vocab -> vocabulary of genes if pre_trained_embeddings used; default: None
    '''

    def __init__(self, infile, fun_infile = None, min_cnt = 1, unigram = True, pre_trained_vocab = [], device = "cpu"):
        self.device = device
        self.min_cnt = min_cnt
        self.read_pairs(infile, pre_trained_vocab)
        self.fun_input = False # boolean flag indicating whether functional inputs are used

        if fun_infile is not None:
            self.fun_input = True
            self.read_fun_pairs(fun_infile)
            self.map_fun_labels()
        self.gene_pairs = torch.tensor(self.gene_pairs)
        self.batch_idx = torch.tensor(0)
        self.unigram_flag = unigram

        if self.unigram_flag: # gene-specific unigram distribution
            self.init_unigram_table()
        else: # common unigram distribution
            self.init_unigram_table_combined()
        if self.device == "cuda":
            self.move_to_cuda()

    # reading the gene pairs
    def read_fun_pairs(self, infile):
        self.fun_pairs = list()
        with open(infile, "r") as f:
            for line in f:
                genes = line.strip().split()
                if genes[0] in self.gene2id and genes[1] in self.gene2id:
                    self.fun_pairs.append((self.gene2id[genes[0]], self.gene2id[genes[1]]))
        print("Number of total functional pairs:", len(self.fun_pairs))
        self.fun_pair_cnt = torch.tensor(len(self.fun_pairs)).float()
        f.close()

    # reading co-expressed gene pairs
    def read_pairs(self, infile, pre_trained_vocab):
        self.gene_pairs = list()
        gene_frequencies = defaultdict(int)
        with open(infile, "r") as f:
            for line in f:
                genes = line.strip().split()
                self.gene_pairs.append((genes[0], genes[1]))
                for gene in genes:
                    gene_frequencies[gene] += 1
            f.close()

        # sorting genes based on their frequencies and assigning IDs correspondingly
        gene_frequencies = {gene: freq for gene, freq in sorted(gene_frequencies.items(), key=lambda item: item[1], reverse = True)}
        self.frequencies = gene_frequencies

        if pre_trained_vocab == [] or len(pre_trained_vocab) == 0:
            self.gene2id = {gene: idx for idx, (gene, freq) in enumerate(self.frequencies.items())}
        else: # if pre-trained embeddings are used then list of genes with embeddings are considered as the final vocabulary
            self.gene2id = {gene: idx for idx, gene in enumerate(pre_trained_vocab)}

        print("*" * 5, "Number of gene pairs before filtering: ", len(self.gene_pairs), "*"*5)
        self.filter_pairs()
        self.vocab_size = len(self.gene2id)
        print("Vocabulary Size (filtered): ", len(self.gene2id), "Minimum Support:", self.min_cnt)

    # this function assigns functional labels to each pair
    def map_fun_labels(self):
        self.fun_pairs = set.intersection(set(self.gene_pairs), self.fun_pairs)
        fun_labels = []
        for pair in self.gene_pairs:
            if pair in self.fun_pairs:
                fun_labels.append(1)
            else:
                fun_labels.append(0)
        print("*" * 5, "Number of Functional gene pairs from the input:", np.sum(fun_labels), "*" * 5)
        self.fun_labels = torch.tensor(fun_labels)

    # filtering the input gene pairs based on minimum support
    def filter_pairs(self):
        print("*"*5, "Filtering Pairs", "*"*5)
        filtered_pairs, fun_labels = [], []
        gene_neighbors = defaultdict(set)
        for pair in self.gene_pairs:
            if pair[0] in self.gene2id and pair[1] in self.gene2id and self.frequencies[pair[0]] >= self.min_cnt and self.frequencies[pair[1]] >= self.min_cnt:
                gid1 = self.gene2id[pair[0]]
                gid2 = self.gene2id[pair[1]]
                filtered_pairs.append((gid1, gid2))
                gene_neighbors[gid1].add(gid2)
                gene_neighbors[gid2].add(gid1)

        self.gene_pairs = filtered_pairs
        self.id2gene = {idx:gene for gene, idx in self.gene2id.items()}
        self.pair_cnt = torch.tensor(len(filtered_pairs)).float()
        self.gene_neighbors = gene_neighbors
        self.frequencies = {gene: self.frequencies[gene] if gene in self.frequencies else 0 for gene in self.gene2id}

    # this function moves all class parameters i.e. input pairs to GPU in one go
    def move_to_cuda(self):
        print("*" * 5, "Moving data to GPU", "*" * 5)
        self.batch_idx = self.batch_idx.to(self.device)
        self.pair_cnt = self.pair_cnt.to(self.device)
        self.gene_pairs = self.gene_pairs.to(self.device)
        self.unigram_tables = self.unigram_tables.to(self.device)
        if self.fun_input:
            self.fun_labels = self.fun_labels.to(self.device)


    # function to generate a batch of gene pairs
    def fetch_batch_pairs(self, batch_size, fun_flag):
        batch_pairs, fun_labels = [], []
        end_idx = self.pair_cnt.item() if self.batch_idx.item() + batch_size >= self.pair_cnt.item() else self.batch_idx.item() + batch_size

        batch_pairs = self.gene_pairs[self.batch_idx.item() : end_idx]
        if fun_flag:
            fun_labels = self.fun_labels[self.batch_idx.item() : end_idx]
        # resetting batch_idx
        self.batch_idx = torch.tensor(0) if self.batch_idx.item() + batch_size > self.pair_cnt.item() else self.batch_idx + batch_size
        return batch_pairs, fun_labels

    # this function resets the batch index to zero
    def reset_batch_index(self):
        self.batch_idx = torch.tensor(0)
        if self.device == "cuda":
            self.batch_idx = self.batch_idx.to(self.device)

    # Unigram sampling tables for negative sample based on gene frequency. Borrowed from https://github.com/Adoni/word2vec_pytorch
    def init_unigram_table_combined(self):
        self.unigram_tables = []
        print("*" * 10, "Generating Unigram distributions", "*" * 10)
        gene_frequencies = np.array(list(self.frequencies.values()))
        ratio = gene_frequencies / self.pair_cnt.item()
        counts = np.round(ratio * self.vocab_size)
        unigram_table = []
        for idx, c in enumerate(counts):
            unigram_table += [idx] * int(c)
        self.unigram_tables = torch.tensor(unigram_table).to(self.device)

    # Unigram distribution generated seperately for each gene based on their co-occurrence frequencies. Neighbors of each gene are excluded
    def init_unigram_table(self):
        self.unigram_tables = []
        print("*" * 10, "Generating Unigram distributions", "*" * 10)
        gene_frequencies = np.array(list(self.frequencies.values()))
        ratio = gene_frequencies / self.pair_cnt.item()
        counts = np.round(ratio * self.vocab_size)
        for gene_id, count in enumerate(gene_frequencies):
            unigram_table = counts.copy()
            neighbors = self.gene_neighbors[gene_id]
            # setting the negative sampling probability to zero for direct neighbors and the gene itself
            for neighbor_id in neighbors:
                unigram_table[neighbor_id] = 0
            unigram_table[gene_id] = 0
            self.unigram_tables.append(unigram_table)
        self.unigram_tables = torch.tensor(self.unigram_tables).float().to(self.device)

    # this function generates negative samples for a given gene ID
    def generate_neg_ids(self, gid, count):
        return torch.randint(0, torch.sum(self.unigram_tables[gid] >0), (count,))

    #  generates negative samples
    def get_neg_samples(self, pos_pairs, count):
        return torch.multinomial(self.unigram_tables, count)[pos_pairs[:,0]]
    
    #  generates negative samples
    def get_neg_samples_combined(self, pos_pairs, count):
        return self.unigram_tables[torch.randint(0, self.unigram_tables.shape[0], (pos_pairs.shape[0]*count,1))].view((pos_pairs.shape[0], count))