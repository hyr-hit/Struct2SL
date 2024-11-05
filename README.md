# Struct2SL: Synthetic lethality prediction based on AlphaFold2 structure information and Graph Convolutional Network
# Abstract
Struct2SL is a synthetic lethal gene pair prediction model based on graph convolutional neural networks, which uses protein information from AlphaFold2 and other sources to enhance the accuracy and versatility of model predictions.


![image](https://github.com/hyr-hit/Struct2SL/blob/main/image/frame.jpg)

# Data
- Protein structure: download from https://alphafold.ebi.ac.uk/
- Protein sequence: download from https://www.uniprot.org/
- PPI network: down from https://string-db.org/
- SL/nonSL: download from https://synlethdb.sist.shanghaitech.edu.cn/home
  
We put the processed data for train and test on [there](https://github.com/lyjps/Struct2GO/tree/master/divided_data)\
We put the Source Data [there](https://github.com/lyjps/Struct2GO/tree/Source_data/Source_data) \

# Usage
## Train the model
Run the ``Struct2SL.py`` script directly to train the model
 ```python
 python Struct2SL.py
 ``` 

## Processing raw data
we provide the proccesed data for training and evaluating directly [there](https://pan.baidu.com/s/1qVr5RuUbg2cDByJMnEVVrw?pwd=uf3s), and then we will explain how to process the raw data.
### Protein struction data
- Download protein structure data and convert the three-dimensional atomic structure of proteins into protein contact maps.
```
cd ./struct/node2vec-master
python predicted_protein_struct2map.py
```
- Extracting structural features.
```
python src/main.py --input ../../data/proteins_edgs --output ../../data/after_node_vec
```
- Get the structural feature matrix
```
cd..
python sort.py
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
python ./node2vec-master/src/main.py --input pre_node2vec_physical.txt --output result.emb.txt
python sort.py
```
- Get the sorted files
- Rename the protein and match its characteristics to obtain the final protein PPI feature ppi_emb for later use
```
python data_pre.py
```

### Protein sequence data
- Download protein sequence data obtain protein sequence features through the Seqvec model.
```
cd ./sequence
python seq2vec.py
```

### Mapping of proteins and genes is achieved to obtain gene feature embedding
- The processing method of sequence features is the same as PPI. Here we provide the processing method of structural features.
```
cd ./protein2gene
python protein2gene.py
```
