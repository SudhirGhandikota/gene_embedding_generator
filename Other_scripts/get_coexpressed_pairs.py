# coding: utf-8
import pandas as pd
import numpy as np
import timeit
import os
import json
import multiprocessing as mp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_arguments():
    parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = 'resolve')
    parser.add_argument('--inpath', default="/",
                        help='Input folder containing GEO files')

    parser.add_argument('--other_infile', default="/",
                        help='File containing set of filtering genes. All gene pairs will be restricted to contain only these set of genes')

    parser.add_argument('--outpath', default="/",
                        help='Output Folder where the final gene pairs have to be stored')

    parser.add_argument('--source', default = "cases",
                        help = "cases/controls")

    parser.add_argument('--corr_threshold', default =0.9,
                        help = 'Correlation Threshold')
    return parser.parse_args()

# this function writes out the gene-context pairs from each dataset
def write_gene_pairs(gene_pairs, filename):
    fout = open(filename,"w")
    cnt = 0
    for pair in gene_pairs:
        # filtering out diagonal entries (i.e same gene pairs)
        if pair[0]==pair[1]: #filtering out
            continue
        fout.write(pair[0]+" "+pair[1]+"\n")
        cnt += 1
    fout.close()
    print("Number of pairs written: ",cnt)

# this function works as a multiprocessing worker which reads in an expression dataset 
# and generates the gene pairs based on the given correlation threshold
def gene_pair_worker(dataset,inpath,source, corr, filtering_genes = []):
    print("*** Dataset: ",dataset," ***")
    filename = ''
    # if both cases and controls used simultaneously
    if source == "both":
        filename = inpath+dataset+"/expr_data.txt"
    else: # if only diseased cases or normal controls used
        filename = inpath+dataset+"/expr_data_"+source+".txt"
    expr_df = pd.read_csv(filename, sep="\t")
    expr_df = expr_df.set_index('Gene_Symbol')
    if len(filtering_genes)>0:
        common_genes = list(set.intersection(set(expr_df.index), set(filtering_genes)))
        expr_df = expr_df.loc[common_genes]

    # computing pairwise correlations between all genes (using numpy arrays i.e. values for faster computation
    corr_mat = pd.DataFrame(np.corrcoef(expr_df.values),index = expr_df.index,columns = expr_df.index)

    # retrieving the upper triangle elements of the correlation matrix
    corr_triu = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))

    # reshaping the upper triangle elements and sorting them
    corr_sorted = corr_triu.stack().sort_values(ascending = False)

    # filtering gene pairs having correlation >= threshold
    pairs = corr_sorted[corr_sorted >= corr].index.tolist()

    print("*"*5," Done: ",dataset," Number of pairs: ", len(pairs),"*"*5)
    return pairs

# this function generates reverse gene pairs
def reverse_pairs(pairs):
    if len(pairs)==0:
        return []
    reversed_pairs = [(pair[1],pair[0]) for pair in pairs]
    return reversed_pairs

# reading the expression dataset, computing pairwise correlations (genes) and writing out gene pairs with r >= corr
# source = cases or controls or both
# inpath -> path to the folder (disease specific for instance) having gene expression datasets
# datasets -> GEO IDs (as folders) containing gene expression files
def get_gene_pairs(inpath, datasets, filtering_genes, corr, source, outpath):
    pool = mp.Pool(mp.cpu_count())

    outfile = os.path.join(outpath, "co_expressed_gene_pairs_" + str(corr) + ".txt") # all pairs
    # running tSNE worker in parallel by sending arguments
    all_pairs = pool.starmap_async(gene_pair_worker, [(dataset, inpath, source, corr, filtering_genes) for dataset in datasets]).get()
    all_pairs = sum(all_pairs, [])
    pool.close()

    print("*"*5, "All pairs: ", len(all_pairs), "*"*5)

    # adding reversed gene pairs
    all_pairs.extend(reverse_pairs(all_pairs))

    # writing out the gene pairs into a text file
    np.savetxt(outfile, np.asarray(all_pairs),fmt = "%s")
    
    print("*"*10," Total gene pairs: ",len(all_pairs),"*"*10)

if __name__ == '__main__':
    args = parse_arguments()
    expr_inpath = args.inpath
    other_infile = args.other_inpath

    filtering_genes = np.loadtxt(other_infile, dtype = str)
    datasets = ["LGRC", "GSE24206", "GSE10667", "GSE53845", "GSE32537", "GSE48149", "GSE134692"]
    start_time = timeit.default_timer()
    get_gene_pairs(expr_inpath, datasets, filtering_genes,
                   float(args.corr_threshold), args.source, args.outpath)
    print("*"*10,"Total time (seconds) - cases: ",timeit.default_timer() - start_time,"*"*10)

# python get_coexpressed_pairs.py --inpath inputdir --other_inpath filtering_genes.txt --source cases --corr_threshold 0.9 --outpath outdir