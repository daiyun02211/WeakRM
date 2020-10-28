import time
import itertools
import numpy as np
import tensorflow as tf
from nets import WSCNN, WSCNNLSTM, WeakRM, WeakRMLSTM
from utils import create_folder

tfk = tf.keras
tfkl = tf.keras.layers
tfkc = tf.keras.callbacks

instance_len = 50
instance_stride = 10
data_name = ''
model_name = 'WeakRM'

data_dir = '' + data_name + '/data_for_mil/'
target_dir = data_dir + 'cp_dir/'
create_folder(target_dir)
checkpoint_filepath = target_dir + model_name + '.h5'

print('loading data')
train_data = np.load(data_dir + 'train_data.npy', allow_pickle=True)
valid_data = np.load(data_dir + 'valid_data.npy', allow_pickle=True)
itest_data = np.load(data_dir + 'test_data.npy', allow_pickle=True)

train_label = np.load(data_dir + 'train_label.npy', allow_pickle=True)
valid_label = np.load(data_dir + 'valid_label.npy', allow_pickle=True)
itest_label = np.load(data_dir + 'test_label.npy', allow_pickle=True)

train_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(train_data, train_label),
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(tf.TensorShape([None, instance_len, 4]),
                                                              tf.TensorShape([None])))
valid_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(valid_data, valid_label),
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(tf.TensorShape([None, instance_len, 4]),
                                                              tf.TensorShape([None])))
itest_dataset = tf.data.Dataset.from_generator(lambda: itertools.zip_longest(itest_data, itest_label),
                                               output_types=(tf.float32, tf.int32),
                                               output_shapes=(tf.TensorShape([None, instance_len, 4]),
                                                              tf.TensorShape([None])))

train_dataset = train_dataset.shuffle(100).batch(1)
valid_dataset = valid_dataset.batch(1)
itest_dataset = itest_dataset.batch(1)

print('creating model')
if isinstance(model_name, str):
    dispatcher = {'WeakRM': WeakRM,
                  'WeakRMLSTM': WeakRMLSTM}
    try:
        model_funname = dispatcher[model_name]
    except KeyError:
        raise ValueError('invalid input')

model = model_funname()

# lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps=len(train_label), decay_rate=0.96)
# opt = tf.keras.optimizers.Adam(learning_rate=lr)
opt = tf.keras.optimizers.Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5)

train_loss = tf.keras.metrics.Mean()
valid_loss = tf.keras.metrics.Mean()
train_auROC = tf.keras.metrics.AUC()
valid_auROC = tf.keras.metrics.AUC()

train_step_signature = [
    tf.TensorSpec(shape=(1, None, instance_len, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(1, 1), dtype=tf.int32)
]


@tf.function(input_signature=train_step_signature)
def train_step(train_seq, train_label):
    with tf.GradientTape() as tape:
        output_probs, _ = model(train_seq, training=True)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true=train_label, y_pred=output_probs)
        total_loss = loss + tf.reduce_sum(model.losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_auROC(y_true=train_label, y_pred=output_probs)


@tf.function(input_signature=train_step_signature)
def valid_step(valid_seq, valid_label):
    inf_probs, _ = model(valid_seq, training=False)
    vloss = tf.keras.losses.BinaryCrossentropy()(y_true=valid_label, y_pred=inf_probs)
    valid_loss(vloss)
    valid_auROC(y_true=valid_label, y_pred=inf_probs)


EPOCHS = 20
current_monitor = np.inf
patient_count = 0

for epoch in tf.range(1, EPOCHS+1):
    train_loss.reset_states()
    valid_loss.reset_states()

    train_auROC.reset_states()
    valid_auROC.reset_states()

    epoch_start_time = time.time()
    for tdata in train_dataset:
        train_step(tdata[0], tdata[1])
    print('Training of epoch {} finished! Time cost is {}s'.format(epoch, round(time.time() - epoch_start_time, 2)))

    valid_start_time = time.time()
    for vdata in valid_dataset:
        valid_step(vdata[0], vdata[1])

    new_valid_monitor = np.round(valid_loss.result().numpy(), 4)
    if new_valid_monitor < current_monitor:
        print('val_loss improved from {} to {}, saving model to {}'.
              format(str(current_monitor), str(new_valid_monitor), checkpoint_filepath))
        model.save_weights(checkpoint_filepath)
        current_monitor = new_valid_monitor
        patient_count = 0
    else:
        print('val_loss did not improved from {}'.format(str(current_monitor)))
        patient_count += 1

    if patient_count == 5:
        break

    template = "Epoch {}, Time Cost: {}s, TL: {}, TROC: {}, VL:{}, VROC: {}"
    print(template.format(epoch, str(round(time.time() - valid_start_time, 2)),
                          str(np.round(train_loss.result().numpy(), 4)),
                          str(np.round(train_auROC.result().numpy(), 4)),
                          str(np.round(valid_loss.result().numpy(), 4)),
                          str(np.round(valid_auROC.result().numpy(), 4)),
                          )
          )

model.load_weights(checkpoint_filepath)

predictions = []
for tdata in itest_dataset:
    pred, _ = model(tdata[0], training=False)
    predictions.append(pred.numpy())

predictions = np.concatenate(predictions, axis=0)
print('Test AUC: ', tf.keras.metrics.AUC()(y_true=itest_label, y_pred=predictions).numpy())
print('Test PRC: ', tf.keras.metrics.AUC(curve='PR')(y_true=itest_label, y_pred=predictions).numpy())