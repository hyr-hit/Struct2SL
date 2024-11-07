from dataclasses import dataclass
import numpy as np
import os
from tqdm import tqdm

for path,dir_list,file_list in os.walk("./after_node_vec"):

    for file in tqdm(file_list):
        data = np.loadtxt(os.path.join(path, file), skiprows=1)
        data = data[data[:,0].argsort()]
        np.savetxt('./proteins_edgs/' + file, data, fmt="%f")
