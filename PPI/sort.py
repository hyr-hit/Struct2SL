from dataclasses import dataclass
import numpy as np
import os
from tqdm import tqdm

for path,dir_list,file_list in os.walk("./test_for_sort"):
    print('1')
    for file in tqdm(file_list):
        data = np.loadtxt(os.path.join(path, file), skiprows=1)
        data = data[data[:,0].argsort()] 
        np.savetxt('./result_for_sort/' + file, data, fmt="%f")
