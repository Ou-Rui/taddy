"""
python 2_prepare_data_ours.py -d wikipedia
"""

from scipy import sparse
import pickle
import time
import os
import math
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, choices=['wikipedia', 'reddit', 'mooc', 'txn_filter'], 
                    default='wikipedia')
# parser.add_argument('--anomaly_per' ,choices=[0.01, 0.05, 0.1] , type=float, default=None)
args = parser.parse_args()
if args.dataset in ('wikipedia', 'reddit', 'mooc'):
  train_per = 0.7
elif args.dataset in ('txn_filter'):
  train_per = 0.34
  
num_snapshot = 20
num_snapshot_train = int(num_snapshot * train_per)
num_snapshot_test = num_snapshot - num_snapshot_train

DATASET_PATH = '/remote-home/ourui/Datasets/{}.csv'.format(args.dataset)

def reindex(src_l, dst_l):
  assert (max(src_l) - min(src_l) + 1 == len(set(src_l)))
  assert (max(dst_l) - min(dst_l) + 1 == len(set(dst_l)))
  print(max(src_l))
  print(max(dst_l))

  max_src = max(src_l)
  new_src_l = list(src_l)   # 创建值拷贝  
  new_dst_l = [x + max_src + 1 for x in dst_l]

  ''' taddy要从0开始编号节点 '''
  # new_src_l += 1    
  # new_dst_l += 1

  print(max(new_src_l))
  print(max(new_dst_l))

  return new_src_l, new_dst_l

def read_csv(path):
  src_l, dst_l, label_l = [], [], []
  feat_l = []
  with open(path) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      src = int(e[0])
      dst = int(e[1])
      # ts = float(e[2])
      label = int(e[3])
      feat = np.array([float(x) for x in e[4:]])

      src_l.append(src)
      dst_l.append(dst)
      label_l.append(label)
      feat_l.append(feat)
  
  src_l, dst_l = reindex(src_l, dst_l)
  return src_l, dst_l, label_l, feat_l

''' main '''
src_l, dst_l, label_l, feat_l = read_csv(DATASET_PATH)
N = max(max(src_l), max(dst_l)) + 1
E = len(src_l)
snapshot_size = math.ceil(E / num_snapshot / 1000) * 1000   # 在千位向上取整
print(f'snapshot_size={snapshot_size}')

snapshot_src_l, snapshot_dst_l, snapshot_label_l, snapshot_weight_l, snapshot_feat_l = [], [], [], [], []
train_adj_list = [ [i] for i in range(N) ]    # 初始添加自环
for i_snapshot in range(num_snapshot):
  start_idx = i_snapshot * snapshot_size
  end_idx = min(start_idx + snapshot_size, E)
  
  snapshot_src = np.array(src_l[start_idx:end_idx], dtype=np.int32)
  snapshot_dst = np.array(dst_l[start_idx:end_idx], dtype=np.int32)
  snapshot_label = np.array(label_l[start_idx:end_idx], dtype=np.int32)
  snapshot_weight = np.ones_like(snapshot_src, dtype=np.int32)
  snapshot_feat = np.array(feat_l[start_idx:end_idx], dtype=np.float32)
  
  snapshot_src_l.append(snapshot_src)
  snapshot_dst_l.append(snapshot_dst)
  snapshot_label_l.append(snapshot_label)
  snapshot_weight_l.append(snapshot_weight)
  snapshot_feat_l.append(snapshot_feat)
  
  if i_snapshot < num_snapshot_train:
    for src, dst in zip(snapshot_src, snapshot_dst):
      train_adj_list[src].append(dst)
      train_adj_list[dst].append(src)   # 双向边

# 对邻接表的每一项排序
for i in range(len(train_adj_list)):
  train_adj_list[i].sort()

with open(f'data/percent/{args.dataset}_{train_per}_0.1.pkl', 'wb') as f:
  pickle.dump(
    (snapshot_src_l,
     snapshot_dst_l,
     snapshot_label_l,
     snapshot_weight_l,
     snapshot_feat_l,
     train_adj_list,
     num_snapshot_train,
     num_snapshot_test,
     N,
     E), f, pickle.HIGHEST_PROTOCOL)

'''
rows: src [S, E_S] 无反向边
cols: dst [S, E_S]
labs: [S, E_S]
weis: [S, E_S]
headtail: 训练集的邻接表, [N, E_N], 添加反向边和自环, 节点号升序
train_size: 训练集的snapshot数量
test_size: 测试集的snapshot数量
n: N
m: E
'''
