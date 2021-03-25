import time
import itertools
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from nets import WSCNN, WSCNNLSTM, WeakRM, WeakRMLSTM


tfk = tf.keras
tfkl = tf.keras.layers
tfdd = tf.data.Dataset
tfkc = tf.keras.callbacks


def train_model(config):
    data_dir = config.data_dir

    print('Loading data ...')
    train_data = np.load(data_dir + 'train_data.npy', allow_pickle=True)
    valid_data = np.load(data_dir + 'valid_data.npy', allow_pickle=True)
    test_data = np.load(data_dir + 'test_data.npy', allow_pickle=True)

    train_label = np.load(data_dir + 'train_label.npy', allow_pickle=True)
    valid_label = np.load(data_dir + 'valid_label.npy', allow_pickle=True)
    test_label = np.load(data_dir + 'test_label.npy', allow_pickle=True)

    train_dataset = tfdd.from_generator(lambda: itertools.zip_longest(train_data, train_label),
                                        output_types=(tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, config.inst_len, 4]),
                                                       tf.TensorShape([None])))
    valid_dataset = tfdd.from_generator(lambda: itertools.zip_longest(valid_data, valid_label),
                                        output_types=(tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, config.inst_len, 4]),
                                                       tf.TensorShape([None])))
    test_dataset = tfdd.from_generator(lambda: itertools.zip_longest(test_data, test_label),
                                        output_types=(tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, config.inst_len, 4]),
                                                       tf.TensorShape([None])))

    train_dataset = train_dataset.shuffle(8).batch(1)
    valid_dataset = valid_dataset.batch(1)
    test_dataset = test_dataset.batch(1)

    print('Creating model ...')
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

    opt = tf.keras.optimizers.Adam(lr=config.lr_init, decay=config.lr_decay)

    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_auROC = tf.keras.metrics.AUC()
    valid_auROC = tf.keras.metrics.AUC()

    train_step_signature = [
        tf.TensorSpec(shape=(1, None, config.inst_len, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(1, 1), dtype=tf.int32)
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(train_seq, train_label):
        with tf.GradientTape() as tape:
            out_prob, _ = model(train_seq, training=True)
            loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_label, y_pred=out_prob)
            total_loss = loss + tf.reduce_sum(model.losses)
            gradients = tape.gradient(total_loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_auROC(y_true=train_label, y_pred=out_prob)

    @tf.function(input_signature=train_step_signature)
    def valid_step(valid_seq, valid_label):
        inf_prob, _ = model(valid_seq, training=False)
        vloss = tf.keras.losses.BinaryCrossentropy()(y_true=valid_label, y_pred=inf_prob)
        valid_loss(vloss)
        valid_auROC(y_true=valid_label, y_pred=inf_prob)

    num_epoch = config.epoch
    current_monitor = np.inf
    patient_count = 0

    for epoch in tf.range(1, num_epoch + 1):
        train_loss.reset_states()
        valid_loss.reset_states()

        train_auROC.reset_states()
        valid_auROC.reset_states()

        epoch_start_time = time.time()
        for tdata in train_dataset:
            train_step(tdata[0], tdata[1])
            num_inst = tf.cast(tf.shape(tdata[0])[1], tf.float32)
            if config.cropping:
                if num_inst > config.crop_threshold:
                    num_crop = tf.reshape(int(0.75 * num_inst), [])
                    input_bag = tf.image.random_crop(tdata[0], [1, num_crop, config.inst_len, 4])
                    train_step(input_bag, tdata[1])
        print('Training of epoch {} finished! Time cost is {}s'.format(epoch, round(time.time() - epoch_start_time, 2)))

        valid_start_time = time.time()
        for vdata in valid_dataset:
            valid_step(vdata[0], vdata[1])

        new_valid_monitor = np.round(valid_loss.result().numpy(), 4)
        if new_valid_monitor < current_monitor:
            if config.cp_path:
                print('val_loss improved from {} to {}, saving model to {}'.
                      format(str(current_monitor), str(new_valid_monitor), config.cp_path))
                model.save_weights(config.cp_path)
            else:
                print('val_loss improved from {} to {}, saving closed'.
                      format(str(current_monitor), str(new_valid_monitor)))
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

    if config.cp_path:
        model.load_weights(config.cp_path)

    if config.eval_after_train:
        predictions = []
        for tdata in test_dataset:
            pred, _ = model(tdata[0], training=False)
            predictions.append(pred.numpy())

        predictions = np.concatenate(predictions, axis=0)
        auc = roc_auc_score(test_label, predictions)
        ap = average_precision_score(test_label, predictions)
        print('Test AUC: ', auc)
        print('Test PRC: ', ap)



