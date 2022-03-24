import os
import random
import networkx as nx
import pickle
import numpy as np
import tarfile
from tqdm import tqdm

import torch

DATASET_PATH = './dataset/'
DATA_NAME = 'nsfnetbw'


def process_graph(filename):
    G = nx.read_gml(filename, destringizer=int)
    n_node = G.number_of_nodes()
    for i in range(n_node): # i: src
        for j in range(n_node): # j: dst
            if j in G[i]:
                # G[i][j] = {0: {'port': 0, 'weight': 1, 'bandwidth': '10kbps'}}
                G[i][j][0]['bandwidth'] = G[i][j][0]['bandwidth'].replace("kbps", "")
    return G, n_node


def create_routing_matrix(G, n_node, routing_file, edges):
    """
    MatrixPath : NxN Matrix
            Matrix where each cell [i,j] contains the path to go from node
            i to node j.
    """
    connections = {}
    max_path_len = 0
    for src in G:
        connections[src] = {}
        for dst in G[src].keys():
            port = G[src][dst][0]['port']
            connections[src][port] = dst
    R = np.loadtxt(routing_file, delimiter=',', dtype=str)
    R = R[:, :-1]
    R = R.astype(int)
    MatrixPath = np.empty((n_node, n_node), dtype=object)
    for src in range(n_node):
        for dst in range(n_node):
            node = src
            path = []
            path_len = 0
            while not node == dst:
                port = R[node][dst]
                next_node = connections[node][port]
                path.append(edges.index((node, next_node)) + 1) # e0预留给PAD
                node = next_node
                path_len += 1
            MatrixPath[src][dst] = path
            if path_len > max_path_len:
                max_path_len = path_len
    return MatrixPath, max_path_len

def create_link_cap(G):
    edges = list(map(lambda x: (x[0], x[1]), G.edges))
    link_cap = [0] # 添加PAD位置
    for e in edges:
        bandwidth = G[e[0]][e[1]][0]['bandwidth']
        link_cap.append(int(bandwidth))
    return edges, link_cap

def mask(MatrixPath, max_path_len, n_node):
    n_path = n_node * (n_node - 1)
    Paths = np.zeros((n_path, max_path_len), dtype=int)
    # mask = np.zeros((n_path, max_path_len), dtype=int)
    ind = 0
    row = []
    col = []
    link_idx = []
    for i in range(n_node):
        for j in range(n_node):
            if not i == j:
                path_len = len(MatrixPath[i][j])
                Paths[ind][:path_len] = MatrixPath[i][j]
                row.extend([ind for _ in range(path_len)])
                col.extend([i for i in range(path_len)])
                link_idx.extend(MatrixPath[i][j])
                ind += 1
    return Paths, row, col, link_idx

def get_result(line, n_node, offset):
    traffic = []
    delay = []
    jitter = []
    for i in range(n_node):
        for j in range(n_node):
            if not i == j:
                traffic.append(float(line[(i*n_node + j) * 3]))
                delay.append(float(line[offset + (i * n_node + j) * 7]))
                jitter.append(float(line[offset + (i * n_node + j) * 7 + 6]))
    return traffic, delay, jitter

def make_pt(file_list, dir_path, pt_path, G, n_node, edges, n_offset, link_cap):
    for file in tqdm(file_list):
        pt_file = file.split('.')[0] + '.pt'
        tar = tarfile.open(os.path.join(dir_path, file), "r:gz")
        dir_info = tar.next()
        routing_file = tar.extractfile(dir_info.name + "/Routing.txt")
        results_file = tar.extractfile(dir_info.name + "/simulationResults.txt")
        MatrixPath, max_path_len = create_routing_matrix(G, n_node, routing_file, edges)
        MatrixPath, row, col, link_idx = mask(MatrixPath, max_path_len, n_node)
        sample_list = []
        for line in results_file:
            line = line.decode().split(",")
            traffic, delay, jitter = get_result(line, n_node, n_offset)
            sample = {
                'package': torch.FloatTensor(traffic),  # [Np, ]
                'bandwidth': torch.FloatTensor(link_cap), # 添加pad [Ne, ]
                'path': torch.LongTensor(MatrixPath),  # edge从1开始编号  np.array: [Np, Lp]
                'row': torch.LongTensor(row),  # list: [Nw, ]
                'col': torch.LongTensor(col),  # list: [Nw, ]
                'link_idx': torch.LongTensor(link_idx),  # edge从1开始编号 list: [Nw, ]
                'delay': torch.FloatTensor(delay),  # list: [Np, ]
                'jitter': torch.FloatTensor(jitter),  # list: [Np, ]
                # 'n_link': n_links,
                # 'n_path': n_paths
            }
            sample_list.append(sample)
        with open(os.path.join(pt_path, pt_file), 'wb') as f:
            pickle.dump(sample_list, f)
        tar.close()


def process(path, data_name, eval_rate):
    dir_path = f"{path}/{data_name}"
    graph_file = f"{dir_path}/graph_attr.txt"
    pt_path = f"{dir_path}/process"
    # 创建数据预处理文件目录
    if not os.path.exists(pt_path):
        os.mkdir(pt_path)
    
    pt_train = f"{pt_path}/train"
    pt_eval = f"{pt_path}/eval"
    if not os.path.exists(pt_train):
        os.mkdir(pt_train)
    if not os.path.exists(pt_eval):
        os.mkdir(pt_eval)

    G, n_node = process_graph(graph_file)
    edges, link_cap = create_link_cap(G)
    # n_paths = n_node * (n_node -1)
    # n_links = len(edges) + 1
    n_offset = n_node * n_node * 3
    _, _, filename = next(os.walk(dir_path))
    tar_files = [f for f in filename if f.endswith(".tar.gz")]
    evaling = int(len(tar_files) * eval_rate)
    random.shuffle(tar_files)
    train_file, eval_file = tar_files[evaling:], tar_files[:evaling]
    print('# process training data...')
    make_pt(train_file, dir_path, pt_train, G, n_node, edges, n_offset, link_cap)
    print('# process evaling data...')
    make_pt(eval_file, dir_path, pt_eval, G, n_node, edges, n_offset, link_cap)
    
            

process(DATASET_PATH, DATA_NAME, 0.2)