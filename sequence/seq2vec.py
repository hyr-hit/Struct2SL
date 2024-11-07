import pickle
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from tqdm import tqdm
# 其他函数保持不变

def get_elmo_model(model_dir, cpu):
    options_file = model_dir / 'options.json'
    weight_file = model_dir / 'weights.hdf5'

    # Check if CUDA is available and desired
    cuda_device = 0 if torch.cuda.is_available() and not cpu else -1
    
    # Load the ELMo model
    elmo = Elmo(options_file=options_file, weight_file=weight_file, num_output_representations=1, dropout=0)
    if cuda_device >= 0:
        elmo = elmo.cuda(cuda_device)
    
    return elmo



def read_fasta( fasta_path, split_char, id_field ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    sequences = dict()
    '''
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper()
    '''

    import pandas as pd

    # 使用pandas读取xlsx文件
    df = pd.read_excel(fasta_path, engine='openpyxl')

    # 创建字典
    # data_dict = pd.Series(df.iloc[:, 3].values,index=df.iloc[:, 0]).to_dict()
    data_dict = pd.Series(df["Sequence"].values, index=df["Entry"]).to_dict()

    return data_dict


def process_embedding(embeddings, per_protein):
    # 适应新版本的Elmo输出格式
    # embeddings['elmo_representations']是一个长度为1的列表，其中包含一个形状为(batch_size, timesteps, embedding_dim)的张量
    embeddings = embeddings['elmo_representations'][0]
    if per_protein:
        # 平均所有时间步长的嵌入
        embeddings = torch.mean(embeddings, dim=1)
    return embeddings.detach().cpu().numpy()

def get_embeddings(seq_dir, emb_path, model_dir, split_char, id_field, cpu, max_chars, per_protein, verbose):
    seq_dict = read_fasta(seq_dir, split_char, id_field)
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]))

    if verbose:
        print('Total number of sequences: {}'.format(len(seq_dict)))

    model = get_elmo_model(model_dir, cpu)

    print('########start seq2vec###########')

    emb_dict = {}
    for identifier, sequence in tqdm(seq_dict):
        # Prepare batch
        character_ids = batch_to_ids([list(sequence)])
        if cpu:
            character_ids = character_ids.cpu()
        else:
            character_ids = character_ids.cuda()

        # Get embeddings
        embeddings = model(character_ids)

        # Process embeddings
        processed_embeddings = process_embedding(embeddings, per_protein)
        emb_dict[identifier] = processed_embeddings

    # Write embeddings to file
    with open(emb_path, 'wb') as f:
        pickle.dump(emb_dict, f)

    if verbose:
        print('Embeddings written to: {}'.format(emb_path))


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            'embedder.py creates ELMo embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.') )
    
    # Path to fasta file (required)
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).' + 
                            'Can also be a directory holding multiple fasta files.')

    # Path for writing embeddings (required)
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A path to a file for saving the created embeddings as NumPy .npz file.')

    # Path to model (optoinal)
    parser.add_argument('--model', type=str, 
                    default=Path.cwd() / 'model',
                    help='A path to a directory holding a pre-trained ELMo model. '+
                        'If the model is not found in this path, it will be downloaded automatically.' +
                        'The file containing the weights of the model must be named weights.hdf5.' + 
                        'The file containing the options of the model must be named options.json')
    
    # Create embeddings for a single protein or for all residues within a protein
    parser.add_argument('--protein', type=bool, 
                    default=False,
                    help='Flag for summarizing embeddings from residue level to protein level ' +
                    'via averaging. Default: False')
    
    # Number of residues within one batch
    parser.add_argument('--batchsize', type=int, 
                    default=15000,
                    help='Number of residues which need to be accumulated before starting batch ' + 
                    'processing. If you encounter an OutOfMemoryError, lower this value. Default: 15000')
    
    # Character for splitting fasta header
    parser.add_argument('--split_char', type=str, 
                    default='|',
                    help='The character for splitting the FASTA header in order to retrieve ' +
                        "the protein identifier. Should be used in conjunction with --id." +
                        "Default: '|' ")
    
    # Field index for protein identifier in fasta header after splitting with --split_char 
    parser.add_argument('--id', type=int, 
                    default=0,
                    help='The index for the uniprot identifier field after splitting the ' +
                        "FASTA header after each symbole in ['|', '#', ':', ' ']." +
                        'Default: 1')
    
    # Whether to use CPU or GPU
    parser.add_argument('--cpu', type=bool, 
                    default=False,
                    help='Flag for using CPU to compute embeddings. Default: False')
    
    # Whether to print some statistics while processing
    parser.add_argument('--verbose', type=bool, 
                    default=True,
                    help='Embedder gives some information while processing. Default: True')
    return parser


def main():
    
    start_time = datetime.datetime.now()
    parser = create_arg_parser()

    args = parser.parse_args()
    seq_dir   = Path( args.input )
    emb_path  = Path( args.output)
    model_dir = Path( args.model )
    split_char= args.split_char
    id_field  = args.id
    cpu_flag  = args.cpu
    per_prot  = args.protein
    max_chars = args.batchsize
    verbose   = args.verbose
    
    get_embeddings( seq_dir, emb_path, model_dir, split_char, id_field, 
                       cpu_flag, max_chars, True, verbose )
    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)

if __name__ == '__main__':
    main()