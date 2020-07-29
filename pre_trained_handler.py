from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from model import EmbeddingGenerator
from gene_pairs import GenePairs
from collections import Counter
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.rcParams['figure.figsize'] = [16, 14]
sns.set(font_scale = 1.5)
sns.set_style("whitegrid")
sns.set_style({'font.family': 'serif'})

# this function loads the saved embeddings for further use
def load_embeddings(filename):
    genes, embeddings = [],[]
    for line in open(filename, "r").readlines()[1:]: # skipping first line with dimension info
        splits = line.split()
        genes.append(splits[0].strip())
        embeddings.append(np.asarray(splits[1:], dtype = "float32"))
    return np.asarray(genes), np.asarray(embeddings)

def parse_arguments():
    parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = 'resolve')
    parser.add_argument('--infile', default = "/",
                        help = 'Input folder containing gene pairs')

    parser.add_argument('--fun_infile', default = "/",
                        help = 'Input folder containing functional gene pairs')

    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension size")

    parser.add_argument('--unsup_loss', default = "neg_sampling", choices = ['neg_sampling','softmax'],
                        help = 'what kind of unsupervised loss to be used')

    parser.add_argument('--neg_size', type=int, default=5,
                        help="Negative sampling size")

    parser.add_argument('--sup_loss', action = 'store_true',
                        help = 'If supervised loss based on functional pairs is needed')

    parser.add_argument('--init_lr', type=float, default = 0.01,
                        help = 'Initial Learning Rate')

    parser.add_argument('--init_emb',
                        help = 'path to file containing pre-trained embeddings')

    parser.add_argument('--batch_size', type=int, default=10000,
                        help="Number of gene pairs in a batch")

    parser.add_argument('--save_every', type=int, default=5,
                        help="Saving gene embeddings after every n epochs")

    parser.add_argument('--outpath', default = "/",
                        help = 'Output Folder where the final embeddings are to be stored')

    parser.add_argument('--num_epochs', type=int, default = 10,
                        help = 'Number of epochs over the entire data')
    return parser.parse_args()

def plot_losses(losses, num_iters, plot_file):
    iterations = list(range(1, num_iters+1))
    plt.plot(iterations, losses, '-ro')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(plot_file)

if __name__ == '__main__':
    args = parse_arguments()

    # setting training parameters for the gene model
    sample_sizes = args.neg_size
    batch_size = args.batch_size
    num_iters = args.num_epochs
    unsup_loss = args.unsup_loss
    init_lr = args.init_lr
    sup_flag = args.sup_loss
    neg_size = args.neg_size
    save_every = args.save_every

    infile = args.infile
    fun_infile = args.fun_infile
    outpath = args.outpath

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unigram_flag = True if unsup_loss in ["neg_sampling", "nce"] else False  # flag indicating if unigram table needs to be constructed

    if args.init_emb is not None:
        pre_vocab, pre_emb_weights = load_embeddings(args.init_emb)
        print(pre_vocab.shape, pre_emb_weights.shape)
        emb_dim = pre_emb_weights.shape[1]
    else:
        # empty variables for pre-training related inputs
        pre_vocab, pre_emb_weights = [], []
        emb_dim = args.emb_dim

    # loading co-expressed and functional gene pairs
    data = GenePairs(infile, fun_infile, min_cnt=5, unigram = unigram_flag, pre_trained_vocab = pre_vocab, device = device)

    print("Number of gene pairs:", data.gene_pairs.shape[0], "Corpus Size:", data.vocab_size,
          "Number of Functional pairs:", torch.sum(data.fun_labels))
    
    outfile = os.path.join(outpath, 'final_embeddings')
    plotfile = os.path.join(outpath, 'loss_plot.png')
    lossfile = os.path.join(outpath, 'losses.txt')
    model = EmbeddingGenerator(data, outfile,
                        emb_dim = emb_dim, batch_size = batch_size,
                        neg_samp_cnt = neg_size, num_iters = num_iters,
                        init_lr = init_lr, device = device,
                        unsup_loss = unsup_loss, sup_flag = sup_flag,
                        pre_emb_weights = pre_emb_weights, save_every=save_every)
    losses = model.train()
    plot_losses(losses, num_iters, plotfile)
    np.savetxt(lossfile, losses)

# --> gene-specific unigram distribution

# nohup python -u pre_trained_handler.py --infile data/cases_gene2vec_pairs_0.85.txt
# --fun_infile data/ppi_gene_pairs.txt
# --init_emb outdir/dim_128_neg_5_iter_10.txt
# --unsup_loss neg_sampling --sup_loss --num_epochs 250 --batch_size 10000 --save_every 10
# --outpath outdir > nohup_pre.out &