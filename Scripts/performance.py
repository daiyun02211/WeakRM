import itertools
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow as tf
from nets import WSCNN, WSCNNLSTM, WeakRM, WeakRMLSTM
from prettytable import PrettyTable

tfk = tf.keras
tfkl = tf.keras.layers
tfkc = tf.keras.callbacks

instance_len = 50
instance_stride = 10

data_name = ''
model_name = 'WeakRM'

data_dir = '' + data_name + '/data_for_mil/'
target_dir = data_dir + 'cp_dir/'
checkpoint_filepath = target_dir + model_name + '.h5'
print('Load weights from:', checkpoint_filepath)

print('loading data')
itest_data = np.load(data_dir + 'test_data.npy', allow_pickle=True)
itest_label = np.load(data_dir + 'test_label.npy', allow_pickle=True)

itest_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(itest_data, itest_label),
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(tf.TensorShape([None, instance_len, 4]),
                                                              tf.TensorShape([None])))
itest_dataset = itest_dataset.batch(1)


accuracy_scores = []
f1_scores = []
recall_scores = []
precision_scores = []
MCCs = []
auROCs = []
auPRCs = []

print('creating model')
if isinstance(model_name, str):
    dispatcher = {'WeakRM': WeakRM,
                  'WeakRMLSTM': WeakRMLSTM}
    try:
        model_funname = dispatcher[model_name]
    except KeyError:
        raise ValueError('invalid input')

model = model_funname()

model(itest_data[0].reshape(1, -1, instance_len, 4).astype(np.float32))
model.load_weights(checkpoint_filepath)

predictions = []
for tdata in itest_dataset:
    pred, _ = model(tdata[0], training=False)
    predictions.append(pred.numpy())

predictions = np.concatenate(predictions, axis=0)

accuracy_scores.append(accuracy_score(y_true=itest_label, y_pred=predictions > 0.5))
f1_scores.append(f1_score(y_true=itest_label, y_pred=predictions > 0.5))
recall_scores.append(recall_score(y_true=itest_label, y_pred=predictions > 0.5))
precision_scores.append(precision_score(y_true=itest_label, y_pred=predictions > 0.5))
MCCs.append(matthews_corrcoef(y_true=itest_label, y_pred=predictions > 0.5))
auROCs.append(roc_auc_score(y_true=itest_label, y_score=predictions))
auPRCs.append(average_precision_score(y_true=itest_label, y_score=predictions))

table = PrettyTable()
column_names = ['Accuracy', 'recall', 'precision', 'f1', 'MCC', 'auROC', 'auPRC']
table.add_column(column_names[0], np.round(accuracy_scores, 4))
table.add_column(column_names[1], np.round(recall_scores, 4))
table.add_column(column_names[2], np.round(precision_scores, 4))
table.add_column(column_names[3], np.round(f1_scores, 4))
table.add_column(column_names[4], np.round(MCCs, 4))
table.add_column(column_names[5], np.round(auROCs, 4))
table.add_column(column_names[6], np.round(auPRCs, 4))

print(table)






