# coding: utf-8
import pandas as pd
import numpy as np
import timeit
import os
import json
import multiprocessing as mp
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils import *

def parse_arguments():
    parser = ArgumentParser(formatter_class = ArgumentDefaultsHelpFormatter, conflict_handler = 'resolve')
    parser.add_argument('--inpath', default="/", help='Input folder containing GEO files')
    parser.add_argument('--other_inpath', default="/", help='Input folder containing other input files')
    parser.add_argument('--outpath', default="/", help='Output Folder where the final gene pairs have to be stored')
    parser.add_argument('--source', required = True, help = "cases/controls")
    parser.add_argument('--corr_threshold', default =0.9, help = 'Correlation Threshold')
    parser.add_argument('--use_PPI', action='store_true', help='Do PPI based filtering')
    parser.add_argument('--use_syn', action='store_true', help='Use gene synonyms of lung-expressed genes')
    return parser.parse_args()

def read_json(filename):
    with open(filename,'r') as f:
        json_data = json.load(f)
    return json_data

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

# this function returns the gene coverage of the gene-context pairs
def get_coverage(gene_pairs,expr_df):
    all_genes = set(expr_df.index.to_list())
    #separating the genes(targets) and their contexts
    genes, contexts = map(set, zip(*gene_pairs))
    vocabulary = set.union(genes,contexts)
    gene_coverage = float(len(vocabulary)/len(all_genes))
    print("Total Number of genes: ",len(all_genes)," Vocabulary Size: ",len(vocabulary),
         "Gene Coverage: ",gene_coverage)

# this function works as a multiprocessing worker which reads in an expression dataset 
# and generates the gene pairs based on the given correlation threshold
def gene_pair_worker(dataset,inpath,source,corr, filtering_genes, use_PPI):
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
    # filtering for gene pairs with experimental evidence (PPI)
    if use_PPI:
        pairs = filter_PPI_pairs(pairs)
    print("*"*5," Done: ",dataset," Number of pairs: ", len(pairs),"*"*5)
    return pairs

# this function converts gene pairs (with symbols) to id pairs (numeric ids of symbols)
def gene2id(gene_pairs):
    if len(gene_pairs)==0:
        return []
    # collecting unique genes i.e. vocabulary
    genes = list(set([gene for pair in gene_pairs for gene in pair]))
    # gene to identifier dictionary
    sym_to_id_dict = {gene:idx for idx, gene in enumerate(genes)}
    # converting gene pairs to id pairs instead
    id_pairs = [(sym_to_id_dict[gene1], sym_to_id_dict[gene2]) for gene1,gene2 in gene_pairs]
    return genes,id_pairs

# this function filters the gene pairs based on minimum support
def filter_PPI_pairs(gene_pairs):
    if len(gene_pairs)==0 :
        return []
    return list(set.intersection(set(gene_pairs), set(PPI_links)))

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
def get_gene_pairs(inpath, datasets, filtering_genes, corr, source, outpath, use_PPI = False):

    dir_name1 = os.path.join(outpath,'gene2vec_pairs_lung_PPI/') if use_PPI else os.path.join(outpath,'gene2vec_pairs_lung/')

    outfile1 = dir_name1 + str(source) + "_gene2vec_pairs_" + str(corr) + ".txt" # all pairs
    json_file1 = dir_name1 + str(source) + "_marker_pair_map_" + str(corr) + ".json" # humanbase marker map
    json_file2 = dir_name1 + str(source) + "_tm_marker_pair_map_" + str(corr) + ".json" # tabulamuris marker map

    pool = mp.Pool(mp.cpu_count())
    # running tSNE worker in parallel by sending arguments
    all_pairs = pool.starmap_async(gene_pair_worker, [(dataset, inpath, source, corr, filtering_genes, use_PPI) for dataset in datasets]).get()
    all_pairs = sum(all_pairs, [])
    pool.close()

    print("*"*5, "All pairs: ", len(all_pairs), "*"*5)

    # adding reversed gene pairs
    all_pairs.extend(reverse_pairs(all_pairs))

    # writing out the gene pairs into a text file
    np.savetxt(outfile1,np.asarray(all_pairs),fmt = "%s")
    
    print("*"*10," Total gene pairs: ",len(all_pairs),"*"*10)

if __name__ == '__main__':
    args = parse_arguments()
    expr_inpath = args.inpath
    other_inpath = args.other_inpath

    if args.use_syn:
        # loading Lung-expressed genes
        print("*"*10, " Reading genes (and synonyms) expressed in Lung Tissue ","*"*10)
        lung_genes = np.loadtxt(os.path.join(other_inpath,'lung_genes_with_syn.txt'), dtype = str)
    else:
        # loading Lung-expressed genes
        print("*" * 10, " Reading genes expressed in Lung Tissue ", "*" * 10)
        lung_genes = np.loadtxt(os.path.join(other_inpath,'lung_genes.txt'), dtype=str)

    print('*'*10, ' Parsing Lung Tissue Cell Markers from CellMarker ','*'*10)
    # lung tissue cell markers downloaded from http://biocc.hrbmu.edu.cn/CellMarker/#
    cell_signatures = read_json(os.path.join(other_inpath,'human_cell_markers.json'))

    cell_cnt = 0
    for cell,signatures in cell_signatures.items():
        cell_cnt += len(signatures)
    print(cell_cnt)
    
    print('*'*10, ' Parsing Lung Tissue Cell Markers from TabulaMuris ','*'*10)
    # lung tissue cell markers based on Tabula Muris Consortium project
    tm_cell_signatures = read_json(os.path.join(other_inpath,'tabula_muris_cell_markers.json'))

    if args.use_PPI:
        ppi_file = os.path.join(other_inpath,"filtered_PPI_links.txt")
        PPI_links = read_gene_pairs(ppi_file)
        # creating a list of lists to list of tuples (for convenience)
        PPI_links = [ tuple([link[0], link[1]]) for link in PPI_links]
        print("*"*10, "Number of PPI Links: ", len(PPI_links), "*"*10)

    datasets = ["LGRC", "GSE24206", "GSE10667", "GSE53845", "GSE32537", "GSE48149", "GSE134692"]
    #datasets = ["LGRC"]
    start_time = timeit.default_timer()
    get_gene_pairs(expr_inpath, datasets, lung_genes, float(args.corr_threshold), args.source, args.outpath, args.use_PPI)
    print("*"*10,"Total time (seconds) - cases: ",timeit.default_timer() - start_time,"*"*10)

# python get_gene_context_pairs_lung.py --inpath /var/www/ghai7c/IPF/ --other_inpath /var/www/ghai7c/IPF/other_gene2vec_inputs --source cases --corr_threshold 0.9 --outpath /var/www/ghai7c/IPF --use_syn
# python get_gene_context_pairs_lung.py --inpath /var/www/ghai7c/IPF/ --other_inpath /var/www/ghai7c/IPF/other_gene2vec_inputs --source cases --corr_threshold 0.85 --outpath /var/www/ghai7c/IPF --use_syn
# python get_gene_context_pairs_lung.py --inpath /var/www/ghai7c/IPF/ --other_inpath /var/www/ghai7c/IPF/other_gene2vec_inputs --source cases --corr_threshold 0.8 --outpath /var/www/ghai7c/IPF --use_syn

# python get_gene_context_pairs_lung.py --inpath /var/www/ghai7c/IPF/ --other_inpath /var/www/ghai7c/IPF/other_gene2vec_inputs --source controls --corr_threshold 0.9 --outpath /var/www/ghai7c/IPF --use_syn
# python get_gene_context_pairs_lung.py --inpath /var/www/ghai7c/IPF/ --other_inpath /var/www/ghai7c/IPF/other_gene2vec_inputs --source controls --corr_threshold 0.8 --outpath /var/www/ghai7c/IPF --use_PPI --use_syn
# python get_gene_context_pairs_lung.py --inpath /var/www/ghai7c/IPF/ --other_inpath /var/www/ghai7c/IPF/other_gene2vec_inputs --source controls --corr_threshold 0.75 --outpath /var/www/ghai7c/IPF --use_PPI --use_syn