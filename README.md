# Struct2SL: Synthetic lethality prediction based on AlphaFold2 structure information and Multilayer Perceptron
# Abstract
Struct2SL is a synthetic lethal gene pair prediction model based on multilayer perceptron, which uses protein information from AlphaFold2 and other sources to enhance the accuracy and versatility of model predictions.


![image](https://github.com/hyr-hit/Struct2SL/blob/main/image/FRAME.jpg)


# Installation
Struct2SL is based on Pytorch and Python
## Train the model
You will need the following packages to run the code:
- python==3.9.19
- torch==2.3.1
- numpy==1.26.4
- pandas==1.3.5
- scikit-learn==1.4.2
- imbalanced-learn==0.12.3
- torchvision==0.19.1
- scipy==1.10.1


# Data
- Protein structure: download from https://alphafold.ebi.ac.uk/
- Protein sequence: download from https://www.uniprot.org/
- PPI network: down from https://string-db.org/
- SL/nonSL: download from https://synlethdb.sist.shanghaitech.edu.cn/v2/#/download
  
We put the processed data for train and test and raw data on [there](https://figshare.com/search?q=Struct2SL)


# Usage
## Train the model
Run the ``Struct2SL.py`` script directly to train the model
 ```python
 python Struct2SL.py
 ``` 

## Processing raw data
we provide the proccesed data for training and evaluating directly [there](https://figshare.com/search?q=Struct2SL), and then we will explain how to process the raw data.
### Protein struction data
- Download protein structure data and convert the three-dimensional atomic structure of proteins into protein contact maps.
```
cd ./struct
python predicted_protein_struct2map.py
```
- Extracting structural features.
```
cd..
python ./node2vec-master/src/main.py --input ./struct/data/proteins_edgs --output ./struct/data/after_node_vec
```
- Get the structural feature matrix
```
cd ./struct
python sort.py
cd..
```

### Protein PPI data
- The original data is 9606.protein.physical.links.v12.0.txt in ./PPI/data
```
cd ./PPI
python pre_pre.py
```
- The output files are id_list.txt & pre_node2vec_physical.txt
The protein names are stored in id_list.txt in the order of appearance, and their graph information (indicated by serial numbers) is stored in pre_node2vec_physical.txt
```
cd..
python ./node2vec-master/src/main.py --input ./PPI/pre_node2vec_physical.txt --output ./PPI/result.emb.txt
```
- Rename the protein and match its characteristics to obtain the final protein PPI feature ppi_emb for later use
```
cd ./PPI
python data_pre.py
cd..
```

### Protein sequence data
- Download the ELMo pre-trained model to /sequence/model
- weigthts: download from [there](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
- options: download from [there](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)

- Download protein sequence data obtain protein sequence features through the Seqvec model.
```
cd ./sequence
python seq2vec.py
```

### Mapping of proteins and genes is achieved to obtain gene feature embedding
- The processing method of sequence features is the same as PPI, which is to replace the key value according to the correspondence between the protein and gene in uniprot. Here we provide the processing method of structural features.
```
python protein2gene.py
```
