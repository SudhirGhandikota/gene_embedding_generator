from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from model import EmbeddingGenerator
from gene_pairs import GenePairs
from collections import Counter
import os
import numpy as np
import pandas as pd
import torch
import sys

def parse_arguments():
    parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = 'resolve')

    parser.add_argument('--infile', type=str, default="/",
                        help='Input folder containing co-expressed gene pairs')

    parser.add_argument('--fun_infile', type=str, default="/",
                        help='Input folder containing functional gene pairs')

    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension size")

    parser.add_argument('--unsup_loss', type = str, default = "neg_sampling", choices=['neg_sampling','Adaptive softmax'],
                        help='what kind of unsupervised loss to be used')

    parser.add_argument('--neg_size', type=int, default=5,
                        help="Negative sampling size")

    parser.add_argument('--unigram', action='store_false',
                        help='If a common unigram distribution should be used. gene-specific distribution used by default')

    parser.add_argument('--sup_loss', action = 'store_true',
                        help = 'If supervised loss based on functional pairs is needed')

    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of training epochs over input gene pairs")

    parser.add_argument('--init_lr', type = float, default = 0.01,
                        help = 'Initial Learning Rate')

    parser.add_argument('--batch_size', type=int, default=10000,
                        help="Number of gene pairs in a batch")

    parser.add_argument('--save_every', type=int, default=2,
                        help="Saving gene embeddings after every n epochs")

    parser.add_argument('--outpath', type=str, default="/",
                        help='Output Folder where the final embeddings are to be stored')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # setting training parameters for the gene model
    dimension = args.emb_dim  # number of dimensions in the final gene embedding
    sample_size = args.neg_size  # size of negative samples

    batch_size = args.batch_size
    num_iters = args.num_epochs

    infile = args.infile
    fun_infile = args.fun_infile
    unsup_loss = args.unsup_loss
    init_lr = args.init_lr
    sup_flag = args.sup_loss
    save_every = args.save_every
    unigram_flag = args.unigram

    # perform checks to make sure all input paths are available
    if not os.path.exists(infile):
        print("*"*10, "The input file does not exist. Please check","*"*10)
        sys.exit(1)

    if sup_flag:
        if not os.path.exists(fun_infile):
            print("*"*10, "The functional input file does not exist. Please check","*"*10)
            sys.exit(1)

    outpath = args.outpath
    loss_fname = os.path.join(outpath, "loss_infos_dim_"+str(dimension)+".txt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # loading co-expressed and functional gene pairs
    data = GenePairs(infile, fun_infile, min_cnt=5, unigram = unigram_flag, device = device)

    print("Number of gene pairs:", data.gene_pairs.shape[0], "Corpus Size:", data.vocab_size)
    if data.fun_input:
        print("Number of Functional pairs:", torch.sum(data.fun_labels))

    print("\n", "*" * 25, "Dimension:", dimension, "Sampling Size:",sample_size,"*" * 25)
    outfile = os.path.join(outpath, "dim_"+str(dimension)+"_neg_"+str(sample_size))

    generator = EmbeddingGenerator(data, outfile,
                        emb_dim = dimension, batch_size = batch_size,
                        neg_samp_cnt=sample_size, num_iters = num_iters,
                        init_lr = init_lr, device = device, unsup_loss = unsup_loss,
                        unigram_flag = unigram_flag, sup_flag = sup_flag, save_every=save_every)
    losses = generator.train()

    loss_info = np.array([[dimension]*num_iters, [sample_size]*num_iters, list(range(1, num_iters+1)), losses]).T
    loss_info = pd.DataFrame(loss_info, columns = ["Dim", "SSize", "Iter", "Normalized Loss"])
    loss_info.to_csv(loss_fname, sep = "\t", index = False, header = True)

# --> gene-specific unigram distribution
# nohup python -u main_handler.py --infile data/cases_gene2vec_pairs_0.85.txt
# --fun_infile data/ppi_gene_pairs.txt --num_epochs 1 --unsup_loss neg_sampling --sup_loss
# --emb_dim 128 --neg_size 5
# --outpath outdir > nohup.out &