from scipy import sparse
import pickle
import time
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['wikipedia', 'reddit', 'mooc', 'txn_filter'], 
                    default='wikipedia')
# parser.add_argument('--anomaly_per' ,choices=[0.01, 0.05, 0.1] , type=float, default=None)
args = parser.parse_args()
if args.dataset in ('wikipedia', 'reddit', 'mooc'):
  train_per = 0.7
elif args.dataset in ('txn_filter'):
  train_per = 0.34



