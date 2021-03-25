import os
import numpy as np


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def embed(sequence, instance_len, instance_stride, zero_pad=True):
    instance_num = (len(sequence)-instance_len)/instance_stride + 1
    gap = int((max(-1, np.floor(instance_num-1)) +
               (np.ceil(instance_num) - np.floor(instance_num))) *
               instance_stride + instance_len - len(sequence))
    left_pad = int(gap / 2)
    right_pad = gap - left_pad
    padded_seq = np.pad(sequence, (left_pad, right_pad), 'constant', constant_values=5)
    bag = []
    for i in np.arange(max(1, np.floor(instance_num)+(gap > 0))):
        instance = padded_seq[int(i*instance_stride):int(i*instance_stride+instance_len)]
        bag.append(instance)
    bag = np.stack(bag).astype(np.int32)
    one_hot_bag = np.eye(5)[bag][:, :, :4].astype(np.float32)
    condition = (bag == 5).reshape(-1, instance_len, 1)
    if zero_pad:
        one_hot_bag = np.where(condition, np.zeros_like(one_hot_bag, dtype=np.float32), one_hot_bag)
    else:
        one_hot_bag = np.where(condition, np.ones_like(one_hot_bag, dtype=np.float32)*(1/4), one_hot_bag)
    return one_hot_bag.astype(np.float32)