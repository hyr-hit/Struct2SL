# Struct2SL: Synthetic lethality prediction based on AlphaFold2 structure information and Graph Convolutional Network
# Abstract
Struct2SL is a synthetic lethal gene pair prediction model based on graph convolutional neural networks, which uses protein information from AlphaFold2 and other sources to enhance the accuracy and versatility of model predictions.


![avatar](/model.png)

# Data
- Protein structure: download from https://alphafold.ebi.ac.uk/
- Protein sequence: download from https://www.uniprot.org/
- PPI network: down from https://string-db.org/
- SL/nonSL: download from https://synlethdb.sist.shanghaitech.edu.cn/home
  
We put the processed data for train and test on [there](https://github.com/lyjps/Struct2GO/tree/master/divided_data)\
We put the Source Data [there](https://github.com/lyjps/Struct2GO/tree/Source_data/Source_data) \
predicted_struct_protein_data.tar.gz、protein_contact_map.tar.gz、struct_feature.tar.gz supplement [there](https://pan.baidu.com/s/15lyLZ2gMwzop50aUennTPQ?pwd=bcqc)\
include:
| File/Folder name                | Description                                              |
| ------------------------------- | -------------------------------------------------------- |
| predicted_struct_protein_data   | Alphafold2 predicted human protein 3D structure datasets.|
| protein_contact_map             | Computed CA-CA protein contact map.                      |
| struct_feature                  | Protein structural features.                             |
| dict_sequence_feature           | Protein sequence features.                               |
| gos_bp.csv                      | GO terms corresponding to all human proteins in the BP branch. |
| gos_mf.csv                      | GO terms corresponding to all human proteins in the MF branch. |
| gos_cc.csv                      | GO terms corresponding to all human proteins in the CC branch. |


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

### Protein sequence data
- Download protein sequence data obtain protein sequence features through the Seqvec model.
```
cd ./data_processing
python seq2vec.py
```

### Fuse protein structure and sequence data and divide the dataset
```
cd ./model
python labels_load.p
cd ./data_processing
python divide_data.py
```
