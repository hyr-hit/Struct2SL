import pandas as pd
import pickle

f = open('id_list.txt')
data = f.readlines()
f.close()
id_feature = {}
id_number = {}
number_feature = {}
for line in data:
    line = line.strip('\n')
    nodes = line.split(' ')
    id_number[int(nodes[1])] = nodes[0]

# print(id_number)

f = open('./result_for_sort/result.emb.txt')
data = f.readlines()
f.close()
for line in data:
    line = line.strip('\n')
    nodes = line.split(' ')
    id_feature[id_number[int(float(nodes[0]))]] = nodes[1:]
    # break

df1 = pd.read_excel('uniprotkb_AND_model_organism_9606_AND_r_2024_04_22.xlsx')

df1 = df1.dropna(axis=0,subset = ['STRING'])

df1['STRING'] = df1['STRING'].str.rstrip(';')

dff = df1[['Entry', 'STRING']] # 取出其中两列

dff.set_index(keys='STRING', inplace=True) # 设置作为key的列为index

dff = dff.T #取它的转置

dic = dff.to_dict(orient='records')[0] #转化成字典，这可能会有多行，导出是一个字典类型的数组，我们取第一项就可以了

print(dic)

result_dic = {}
for id in dic.keys():
    if id in id_feature.keys():
        print(id)
        result_dic[dic[id]] = id_feature[id]

with open('./PPI_feature/ppi_emb','wb')as f:
    pickle.dump(result_dic,f)
