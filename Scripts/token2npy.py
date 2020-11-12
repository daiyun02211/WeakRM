import os
import numpy as np
import rpy2.robjects as ro
from utils import embed

np.random.seed(323)

readRDS = ro.r['readRDS']

data_dir = ''

instance_len = 50
instance_stride = 10

train_seq = readRDS(data_dir + 'train_seq.rds')
train_seq = np.asarray(train_seq)
train_seq = [np.asarray(seq) for seq in train_seq]

valid_seq = readRDS(data_dir + 'valid_seq.rds')
valid_seq = np.asarray(valid_seq)
valid_seq = [np.asarray(seq) for seq in valid_seq]

test_seq = readRDS(data_dir + 'test_seq.rds')
test_seq = np.asarray(test_seq)
test_seq = [np.asarray(seq) for seq in test_seq]

target_dir = data_dir + 'data_for_weak/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

train_bags = []
for seq in train_seq:
    ont_hot_bag = embed(seq, instance_len, instance_stride, zero_pad=False)
    train_bags.append(ont_hot_bag)

train_bags = np.asarray(train_bags)

valid_bags = []
for seq in valid_seq:
    ont_hot_bag = embed(seq, instance_len, instance_stride, zero_pad=False)
    valid_bags.append(ont_hot_bag)

valid_bags = np.asarray(valid_bags)

test_bags = []
for seq in test_seq:
    ont_hot_bag = embed(seq, instance_len, instance_stride, zero_pad=False)
    test_bags.append(ont_hot_bag)

test_bags = np.asarray(test_bags)

np.save(target_dir + 'train_data.npy', train_bags)
np.save(target_dir + 'valid_data.npy', valid_bags)
np.save(target_dir + 'test_data.npy', test_bags)

train_label = readRDS(data_dir + 'train_label.rds')
train_label = np.asarray(train_label)

valid_label = readRDS(data_dir + 'valid_label.rds')
valid_label = np.asarray(valid_label)

test_label = readRDS(data_dir + 'test_label.rds')
test_label = np.asarray(test_label)

np.save(target_dir + 'train_label.npy', train_label.reshape(-1, 1).astype(np.int32))
np.save(target_dir + 'valid_label.npy', valid_label.reshape(-1, 1).astype(np.int32))
np.save(target_dir + 'test_label.npy', test_label.reshape(-1, 1).astype(np.int32))