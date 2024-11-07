f = open('9606.protein.physical.links.v12.0.txt')
data = f.readlines()
f.close()
lines = []
nodes_list = {}
id_list = []
counter = -1
for line in data:
    if counter < 0:
        counter += 1
        continue
    line = line.strip('\n')
    nodes = line.split(' ')
    ids = [1,2]
    ids[0] = nodes[0]
    ids[1] = nodes[1]
    nodes[0] = nodes[0].strip('9606.ENSP')
    nodes[1] = nodes[1].strip('9606.ENSP')
    if nodes[0] not in nodes_list.keys():
        nodes_list[nodes[0]] = counter
        id_list.append(str(ids[0]) + ' ' + str(counter) + '\n')
        counter += 1
    if nodes[1] not in nodes_list.keys():
        nodes_list[nodes[1]] = counter
        id_list.append(str(ids[1]) + ' ' + str(counter) + '\n')
        counter += 1
    new_line = str(nodes_list[nodes[0]]) + ' ' + str(nodes_list[nodes[1]]) + ' ' + str(nodes[2]) + '\n'
    lines.append(new_line)

with open('id_list.txt', 'w') as f:
    f.writelines(id_list)
    
    
with open('pre_node2vec_physical.txt', 'w') as f:
    f.writelines(lines)