import itertools
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from nets import WSCNN, WSCNNLSTM, WeakRM, WeakRMLSTM
from prettytable import PrettyTable


def eval_model(config):
    test_data = np.load(config.data_dir + 'test_data.npy', allow_pickle=True)
    test_label = np.load(config.data_dir + 'test_label.npy', allow_pickle=True)

    test_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(test_data, test_label),
                                                  output_types=(tf.float32, tf.int32),
                                                  output_shapes=(tf.TensorShape([None, config.inst_len, 4]),
                                                                 tf.TensorShape([None])))
    test_dataset = test_dataset.batch(1)

    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    MCCs = []
    auROCs = []
    auPRCs = []

    print('creating model ...')
    if isinstance(config.model_name, str):
        dispatcher = {'WeakRM': WeakRM,
                      'WeakRMLSTM': WeakRMLSTM,
                      'WSCNN': WSCNN,
                      'WSCNNLSTM': WSCNNLSTM}
        try:
            network = dispatcher[config.model_name]
        except KeyError:
            raise ValueError('invalid model name!')

    if config.model_name.startswith('Weak'):
        model = network()
    else:
        model = network(merging=config.merging)

    model(test_data[0].reshape(1, -1, config.inst_len, 4).astype(np.float32))

    model.load_weights(config.cp_path)

    predictions = []
    for tdata in test_dataset:
        pred, _ = model(tdata[0], training=False)
        predictions.append(pred.numpy())

    predictions = np.concatenate(predictions, axis=0)

    thres = config.threshold
    accuracy_scores.append(accuracy_score(y_true=test_label, y_pred=predictions > thres))
    f1_scores.append(f1_score(y_true=test_label, y_pred=predictions > thres))
    recall_scores.append(recall_score(y_true=test_label, y_pred=predictions > thres))
    precision_scores.append(precision_score(y_true=test_label, y_pred=predictions > thres))
    MCCs.append(matthews_corrcoef(y_true=test_label, y_pred=predictions > thres))
    auROCs.append(roc_auc_score(y_true=test_label, y_score=predictions))
    auPRCs.append(average_precision_score(y_true=test_label, y_score=predictions))

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
