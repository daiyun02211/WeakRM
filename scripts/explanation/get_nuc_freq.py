import numpy as np
import rpy2.robjects as ro

np.random.seed(323)

readRDS = ro.r['readRDS']

data_name = 'Lung'
data_dir = '/home/daiyun/DATA/MLab/m6acsvar/processed_data/' + data_name + '/'

test_seq = readRDS(data_dir + 'test_seq.rds')
test_seq = np.asarray(test_seq)
test_seq = [np.asarray(seq) for seq in test_seq]

nuc_counts = []
for seq in test_seq:
    _, count = np.unique(seq, return_counts=True)
    nuc_counts.append(count.reshape(1, 4))

nuc_counts = np.concatenate(nuc_counts)
nuc_freq = np.sum(nuc_counts, axis=0)
nuc_freq = nuc_freq / np.sum(nuc_freq)
print(nuc_freq)