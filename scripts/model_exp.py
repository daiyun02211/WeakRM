import itertools
import numpy as np
import tensorflow as tf
from nets import WeakRM, WeakRMLSTM
from utils import create_folder
from explanation.exp_utils import fixed_ig, dishuffle_ig
from explanation.visualization import plot_weights
from matplotlib import pyplot as plt

tfk = tf.keras
tfkl = tf.keras.layers
tfkc = tf.keras.callbacks

data_name = ''
model_name = 'WeakRM'
# currently only fixed and dishuffle are supported
ref = 'fixed'

data_dir = '' + data_name + '/data_for_mil/'
visual_dir = data_dir + 'visual/'
create_folder(visual_dir)
target_dir = data_dir + 'exp/'
create_folder(target_dir)
cp_dir = data_dir + 'cp_dir/'
checkpoint_filepath = cp_dir + model_name + '.h5'
if ref == 'dishuffle':
    ref_seq = np.load(target_dir + 'shuffled_ref_bag.npy', allow_pickle=True).item()
target_dir = target_dir + model_name + '/'

print('loading data')
itest_data = np.load(data_dir + 'test_data.npy', allow_pickle=True)
itest_label = np.load(data_dir + 'test_label.npy', allow_pickle=True)

instance_len = 40
instance_stride = 5

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
print('Load weights from:', checkpoint_filepath)
model.load_weights(checkpoint_filepath)

predictions = []
weights = []
for i in np.arange(len(itest_data)):
    y_pred, a_weights = model(itest_data[i].reshape(1, -1, instance_len, 4).astype(np.float32), training=False)
    predictions.append(y_pred)
    weights.append(a_weights)
predictions = np.concatenate(predictions, axis=0)

inds = [i[0] for i in sorted(enumerate(predictions), key=lambda x:x[1], reverse=True)
       if (predictions[i[0]] > 0.5) & (itest_label[i[0]] == 1)]

ig_scores = []
hype_scores = []
one_hot_data = []

if ref == 'fixed':
    inds = inds[:10]
    for idx, ind in enumerate(inds):
        vinst_idx = np.argmax(weights[ind], axis=1)[0]
        ig_score, hype_score = fixed_ig(itest_data[ind], model, freq=False)
        plot_weights(ig_score[vinst_idx], highlight={})
        plt.savefig(visual_dir + str(idx+1) + '.pdf', bbox_inches='tight', pad_inches=0, dpi=350)
        # plt.show()
        plt.close()
        # plot_weights(hype_score[vinst_idx])
        # plt.show()
        print('{}/{} finished !'.format(idx + 1, len(inds)))
elif ref == 'dishuffle':
    for idx, ind in enumerate(inds):
        vinst_idx = np.argmax(weights[ind], axis=1)[0]
        ig_score, hype_score = dishuffle_ig(itest_data[ind], model, ref_bag=ref_seq[str(ind)],
                                            shuffle_times=20, ig_step=20)
        # plot_weights(ig_score[vinst_idx], highlight={})
        # plt.show()

        ig_scores.append(ig_score[vinst_idx][np.newaxis, ...])
        hype_scores.append(hype_score[vinst_idx][np.newaxis, ...])
        one_hot_data.append(itest_data[ind][vinst_idx][np.newaxis, ...])

        print('{}/{} finished !'.format(idx + 1, len(inds)))
    ig_scores = np.concatenate(ig_scores, axis=0)
    hype_scores = np.concatenate(hype_scores, axis=0)
    one_hot_data = np.concatenate(one_hot_data, axis=0)
    print(one_hot_data.shape)
    np.save(target_dir + 'shuffle_task_to_scores.npy', ig_scores)
    np.save(target_dir + 'shuffle_task_to_hyp_scores.npy', hype_scores)
    np.save(target_dir + 'shuffle_onehot_data.npy', one_hot_data)
else:
    raise('Current your selected reference method is not supported!')





