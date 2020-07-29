# A multi-context feature learning approach to identify disease-specific gene neighborhoods

A network-based representation learning framework which uses both co-expressed and functional gene pairs to learn continuous gene representations. 

## Training the model
Our model requires two sets of input gene pairs (co-expressed and functionally related). Both the input files should be space-seperated format as shown below:
```
SERPINB3 SERPINB4
ARMC3 CFAP52
C9orf24 CFAP52
SNTN CFAP52
```
Then our feature learning methodology can be implemented using the below syntax:
```
python main_handler.py --infile co_expressed_pairs.txt --fun_infile functional_pairs.txt
```
We include both sets of inputs used in our manuscript in the *data* folder of the repository.
To see the entire list of parameters/options:
```
python main_handler.py --help
```

## Using pre-trained embeddings
Our model can also be trained using a set of pre-trained gene embeddings using the below command.
```
python pre_trained_handler.py --infile co_expressed_pairs.txt --fun_infile functional_pairs.txt 
                              --init_emb embeddings.txt
```
The input gene embeddings should be provided in a space-delimited file with the first column containing the gene identified or a symbol. The remaining N columns in each row represent the N-dimensional gene representation.

These pre-trained embeddings could be from an earlier training iteration or representations of genes learned within the same disease context.

## Code dependencies
Our method was tested in Python 3.7. The required dependencies or packages include [PyTorch](https://pytorch.org/), [numpy](http://www.numpy.org/), [pandas](https://pandas.pydata.org/).
