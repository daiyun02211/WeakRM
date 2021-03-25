import argparse
import numpy as np
from utils import create_folder, embed

parser = argparse.ArgumentParser(description="Convert token to bags")
parser.add_argument('--input_dir', default='../Data/m7G/', type=str,
                    help='Path to token directory')
parser.add_argument('--output_dir', default='../Data/m7G/processed/', type=str,
                    help='Path to processed data directory')
parser.add_argument('--len', default='50', type=int,
                    help='Instance length')
parser.add_argument('--stride', default='10', type=int,
                    help='Instance stride')

args = parser.parse_args()

data_dir = args.input_dir
target_dir = args.output_dir
create_folder(target_dir)

inst_len = args.len
inst_stride = args.stride

train_token = np.load(data_dir + 'train_token.npy', allow_pickle=True)
valid_token = np.load(data_dir + 'valid_token.npy', allow_pickle=True)
test_token = np.load(data_dir + 'test_token.npy', allow_pickle=True)

train_label = np.load(data_dir + 'train_label.npy', allow_pickle=True)
valid_label = np.load(data_dir + 'valid_label.npy', allow_pickle=True)
test_label = np.load(data_dir + 'test_label.npy', allow_pickle=True)

train_bags = []
for seq in train_token:
    ont_hot_bag = embed(seq, inst_len, inst_stride)
    train_bags.append(ont_hot_bag)

train_bags = np.asarray(train_bags)

valid_bags = []
for seq in valid_token:
    ont_hot_bag = embed(seq, inst_len, inst_stride)
    valid_bags.append(ont_hot_bag)

valid_bags = np.asarray(valid_bags)

test_bags = []
for seq in test_token:
    ont_hot_bag = embed(seq, inst_len, inst_stride)
    test_bags.append(ont_hot_bag)

test_bags = np.asarray(test_bags)

np.save(target_dir + 'train_data.npy', train_bags)
np.save(target_dir + 'valid_data.npy', valid_bags)
np.save(target_dir + 'test_data.npy', test_bags)

np.save(target_dir + 'train_label.npy', train_label)
np.save(target_dir + 'valid_label.npy', valid_label)
np.save(target_dir + 'test_label.npy', test_label)
