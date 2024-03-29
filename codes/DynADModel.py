import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_bert import BertPreTrainedModel
from codes.BaseModel import BaseModel

import time
import os
import pickle
import numpy as np

from sklearn import metrics
from codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL


class DynADModel(BertPreTrainedModel):
  learning_record_dict = {}
  lr = 0.001
  weight_decay = 5e-4
  max_epoch = 500
  spy_tag = True

  load_pretrained_path = ''
  save_pretrained_path = ''

  def __init__(self, config, args):
    super(DynADModel, self).__init__(config, args)
    self.args = args
    self.config = config
    self.transformer = BaseModel(config)
    self.cls_y = torch.nn.Linear(config.hidden_size, 1)
    self.weight_decay = config.weight_decay
    self.init_weights()

  def forward(self, raw_embedding, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):

    outputs = self.transformer(raw_embedding, init_pos_ids, hop_dis_ids, time_dis_ids)

    sequence_output = 0
    for i in range(self.config.k+1):
      sequence_output += outputs[0][:,i,:]
    sequence_output /= float(self.config.k+1)

    output = self.cls_y(sequence_output)

    return output

  def batch_cut(self, idx_list):
    batch_list = []
    for i in range(0, len(idx_list), self.config.batch_size):
      batch_list.append(idx_list[i:i + self.config.batch_size])
    return batch_list

  def evaluate(self, trues, preds):
    aucs = {}
    for snap in range(len(self.data['snap_test'])):
      auc = metrics.roc_auc_score(trues[snap],preds[snap])
      aucs[snap] = auc

    trues_full = np.hstack(trues)
    preds_full = np.hstack(preds)
    auc_full = metrics.roc_auc_score(trues_full, preds_full)
    
    return aucs, auc_full

  def generate_embedding(self, edges, negative_flag=False):
    num_snap = len(edges)
    # WL_dict = compute_WL(self.data['idx'], np.vstack(edges[:7]))
    WL_dict = compute_zero_WL(self.data['idx'],  np.vstack(edges[:7]))    # 7是什么鬼, 似乎是train_snapshot的数量
    batch_hop_dicts = compute_batch_hop(self.data['idx'], edges, num_snap, self.data['S'], self.config.k, self.config.window_size)
    
    use_raw_feat = self.config.dataset in ('wikipedia', 'reddit', 'mooc', 'txn_filter')
    raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
        dicts_to_embeddings(self.data['X'], batch_hop_dicts, WL_dict, num_snap, 
                            use_raw_feat=use_raw_feat, negative_flag=negative_flag)
    return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

  def negative_sampling(self, edges):
    '''
    edges: [S_train, [E_S, 2]] 训练集的边
    '''
    negative_edges = []
    node_list = self.data['idx']
    num_node = node_list.shape[0]
    for snap_edge in edges:
      num_edge = snap_edge.shape[0]

      negative_edge = snap_edge.copy()
      fake_idx = np.random.choice(num_node, num_edge)
      fake_position = np.random.choice(2, num_edge).tolist()
      fake_idx = node_list[fake_idx]
      negative_edge[np.arange(num_edge), fake_position] = fake_idx

      negative_edges.append(negative_edge)
    return negative_edges

  def train_model(self, max_epoch):

    optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    embedding_filename = f'./data/embedding/{self.args.dataset}.pkl'
    raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = None, None, None, None, None
    if not os.path.exists(embedding_filename):
      print(f'Embedding file not found. Generating embeddings....')
      raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges']) # [S, [E_S, 14]] 第一项是None, int不是E_S??
      print(f'Generate Done')
      with open(embedding_filename, 'wb') as f:
        pickle.dump((raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings), f, pickle.HIGHEST_PROTOCOL)
      print(f'Dump Done!')
    else:
      print(f'Loading Embedding {embedding_filename}...')
      with open(embedding_filename, 'rb') as f:
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = pickle.load(f)
      print(f'Loaded!')
        
    self.data['raw_embeddings'] = None

    ns_function = self.negative_sampling
    
    # Epoch Loop
    for epoch in range(max_epoch):
      t_epoch_begin = time.time()

      # -------------------------
      embedding_neg_filename = f'./data/embedding/{self.args.dataset}_neg_{self.args.train_per}_{self.args.anomaly_per}.pkl'
      negatives = None
      raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = None, None, None, None, None
      if not os.path.exists(embedding_neg_filename):
        print(f'Negative Embedding file not found. Generating embeddings....')
        negatives = ns_function(self.data['edges'][:max(self.data['snap_train']) + 1])
        raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = self.generate_embedding(negatives)    
        print(f'Generate Done')
        with open(embedding_neg_filename, 'wb') as f:
          pickle.dump((negatives, raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg),
                      f, pickle.HIGHEST_PROTOCOL)
        print(f'Dump Done!')
      else:
        print(f'Loading Negative Embedding {embedding_neg_filename}...')
        with open(embedding_neg_filename, 'rb') as f:
          negatives, raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = pickle.load(f)
        print(f'Loaded!')
          

      self.train()

      loss_train = 0
      # Train Snap Loop
      for snap in self.data['snap_train']:
        if wl_embeddings[snap] is None:
          continue
        # raw_embedding_pos = raw_embeddings[snap]
        int_embedding_pos = int_embeddings[snap]    # [E, 14] ???14又是什么 wiki [1668, 14]????
        hop_embedding_pos = hop_embeddings[snap]    # 
        time_embedding_pos = time_embeddings[snap]
        y_pos = self.data['y'][snap].float()            # 正样本标签

        # raw_embedding_neg = raw_embeddings_neg[snap]
        int_embedding_neg = int_embeddings_neg[snap]  # wiki [7993, 14], uci [997, 14]???
        hop_embedding_neg = hop_embeddings_neg[snap] 
        time_embedding_neg = time_embeddings_neg[snap]
        y_neg = torch.ones(int_embedding_neg.size()[0])

        # raw_embedding = torch.vstack((raw_embedding_pos, raw_embedding_neg))
        int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
        hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
        time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
        y = torch.hstack((y_pos, y_neg))
        
        if self.config.dataset in ('wikipedia', 'reddit', 'mooc', 'txn', 'txn_filter'):
          raw_embedding_pos = raw_embeddings[snap]
          raw_embedding_neg = raw_embeddings_neg[snap]
          raw_embedding = torch.vstack((raw_embedding_pos, raw_embedding_neg))
        else:
          raw_embedding = None

        optimizer.zero_grad()

        output = self.forward(raw_embedding, int_embedding, hop_embedding, time_embedding).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, y)
        loss.backward()
        optimizer.step()

        loss_train += loss.detach().item()
      # End Train Snap Loop
      
      loss_train /= len(self.data['snap_train']) - self.config.window_size + 1
      print('Epoch: {}, loss:{:.4f}, Time: {:.4f}s'.format(epoch + 1, loss_train, time.time() - t_epoch_begin))

      if ((epoch + 1) % self.args.print_feq) == 0:
        self.eval()
        preds = []
        for snap in self.data['snap_test']:
          int_embedding = int_embeddings[snap]
          hop_embedding = hop_embeddings[snap]
          time_embedding = time_embeddings[snap]
          
          if self.config.dataset in ('wikipedia', 'reddit', 'mooc', 'txn', 'txn_filter'):
            raw_embedding = raw_embeddings[snap]
          else:
            raw_embedding = None

          with torch.no_grad():
            output = self.forward(raw_embedding, int_embedding, hop_embedding, time_embedding, None)
            output = torch.sigmoid(output)
          pred = output.squeeze().numpy()
          preds.append(pred)

        y_test = self.data['y'][min(self.data['snap_test']):max(self.data['snap_test'])+1]
        y_test = [y_snap.numpy() for y_snap in y_test]

        aucs, auc_full = self.evaluate(y_test, preds)

        for i in range(len(self.data['snap_test'])):
          print("Snap: %02d | AUC: %.4f" % (self.data['snap_test'][i], aucs[i]))
        print('TOTAL AUC:{:.4f}'.format(auc_full))
    # End Epoch Loop
    
    
  def run(self):
    self.train_model(self.max_epoch)
    return self.learning_record_dict