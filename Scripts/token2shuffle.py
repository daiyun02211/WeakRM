import numpy as np
import rpy2.robjects as ro
from explanation.exp_utils import dinuc_shuffle

np.random.seed(323)

readRDS = ro.r['readRDS']

data_name = ''
data_dir = '' + data_name + '/'
target_dir = data_dir + 'data_for_mil/exp/'
test_seq = readRDS(data_dir + 'test_seq.rds')
test_seq = np.asarray(test_seq)
test_seq = [np.asarray(seq) for seq in test_seq]

test_label = readRDS(data_dir + 'test_label.rds')
test_label = np.asarray(test_label)

shuffle_times = 20

shuffle_seqs = []
for i in np.arange(len(test_seq)):

    shuffle_refs = []
    for j in np.arange(shuffle_times):
        cseq = test_seq[i].astype(np.int32)
        shuffle_refs.append(dinuc_shuffle(np.eye(4)[cseq-1])[np.newaxis, ...])
    shuffle_seqs.append(np.concatenate(shuffle_refs, axis=0))

instance_len = 40
instance_stride = 5


def embed(sequence, instance_len, instance_stride):
    instance_num = int((sequence.shape[1] - instance_len)/instance_stride) + 1
    bag = []
    for i in range(instance_num):
        instance = sequence[:, i*instance_stride:i*instance_stride+instance_len, :]
        bag.append(instance[:, np.newaxis, ...])
    bag = np.concatenate(bag, axis=1).astype(np.float32)
    return bag


shuffle_bags = {}
for idx, seq in enumerate(shuffle_seqs):
    shuffle_bags[str(idx)] = embed(seq, instance_len, instance_stride)

np.save(target_dir + 'shuffled_ref_bag.npy', shuffle_bags)