import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid

tfk = tf.keras
tfkl = tf.keras.layers
tfkc = tf.keras.callbacks


class Noisyand(tf.keras.layers.Layer):

    def __init__(self, a=7.5, **kwargs):
        super(Noisyand, self).__init__(**kwargs)
        self.a = a

    def build(self, input_shape):
        self.b = self.add_weight(shape=(input_shape[-1], ),
                                 initializer=tfk.initializers.RandomUniform(minval=0, maxval=1),
                                 constraint=tfk.constraints.MinMaxNorm(min_value=0.0, max_value=1.0),
                                 trainable=True)

    def call(self, inputs, training=True, mask=None):
        if len(inputs.shape) == 4:
            inputs = tf.squeeze(inputs, 2)
        part1 = sigmoid((tf.reduce_mean(inputs, axis=1) - self.b) * self.a) - sigmoid(-self.a * self.b)
        part2 = sigmoid(self.a * (1 - self.b)) - sigmoid(-self.a * self.b)
        return part1 / part2

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class WSCNN(tf.keras.Model):

    def __init__(self, instance_len=50, merging='MAX', a=7.5):
        super(WSCNN, self).__init__()

        assert merging in ['MAX', 'AVG', 'NOISY']

        self.conv1 = tfkl.Conv2D(16, (1, 15), padding='same',
                                 activation='relu', kernel_regularizer=l2(0.005))
        self.conv2 = tfkl.Conv2D(32, (1, 1), padding='same',
                                 activation='relu', kernel_regularizer=l2(0.005))
        self.conv3 = tfkl.Conv2D(1, (1, 1), padding='same',
                                 activation='sigmoid', kernel_regularizer=l2(0.005))
        self.dropout = tfkl.Dropout(0.5)
        self.pool1 = tfkl.MaxPool2D(pool_size=(1, instance_len))

        if merging == 'MAX':
            self.pool2 = tfkl.GlobalMaxPooling2D()
        elif merging == 'AVG':
            self.pool2 = tfkl.GlobalAveragePooling2D()
        elif merging == 'NOISY':
            self.pool2 = Noisyand(a=a)

    def call(self, inputs, training=True, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.conv3(x)
        out = self.pool2(x)
        return out


class WSCNNLSTM(tf.keras.Model):

    def __init__(self, merging='MAX', a=7.5):
        super(WSCNNLSTM, self).__init__()

        assert merging in ['MAX', 'AVG', 'NOISY']

        self.conv1 = tf.keras.layers.Conv2D(16, (1, 15), padding='same', activation='relu',
                                            kernel_regularizer=l2(0.005))
        self.conv2 = tf.keras.layers.Conv1D(1, 1, padding='same', activation='sigmoid',
                                            kernel_regularizer=l2(0.005))
        self.lstm = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
        self.dropout1 = tfkl.Dropout(0.2)
        self.dropout2 = tfkl.Dropout(0.5)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 5))
        if merging == 'MAX':
            self.pool2 = tfkl.GlobalMaxPooling1D()
        elif merging == 'AVG':
            self.pool2 = tfkl.GlobalAveragePooling1D()
        elif merging == 'NOISY':
            self.pool2 = Noisyand(a=a)

    def call(self, inputs, training=True, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.lstm(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.conv2(x)
        out = self.pool2(x)

        return out


class WeakRM(tf.keras.Model):

    def __init__(self):
        super(WeakRM, self).__init__()

        self.conv1 = tfkl.Conv1D(32, 15, padding='same', activation='relu')
        self.conv2 = tfkl.Conv1D(16, 5, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.005))
        self.dropout = tfkl.Dropout(0.25)
        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_bag = tf.squeeze(inputs, axis=0)

        inst_conv1 = self.conv1(input_bag)
        inst_pool1 = self.pool1(inst_conv1)
        inst_pool1 = self.dropout(inst_pool1, training=training)

        inst_conv2 = self.conv2(inst_pool1)

        inst_features = tfkl.Flatten()(inst_conv2)

        attention_vmatrix = self.att_v(inst_features)
        attention_umatrix = self.att_u(inst_features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, inst_features)

        bag_probability = self.classifier(bag_features)

        return bag_probability, gated_attention


class WeakRMLSTM(tf.keras.Model):

    def __init__(self):
        super(WeakRMLSTM, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(16, (1, 15), padding='same', activation='relu')
        self.lstm = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)))
        self.dropout = tfkl.Dropout(0.25)
        self.pool1 = tfkl.MaxPool2D(pool_size=(1, 2))

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_bag = inputs
        inst_conv1 = self.conv1(input_bag)
        inst_pool1 = self.pool1(inst_conv1)
        inst_pool1 = self.dropout(inst_pool1, training=training)

        inst_conv2 = self.lstm(inst_pool1)

        inst_features = tf.squeeze(inst_conv2, axis=0)

        attention_vmatrix = self.att_v(inst_features)
        attention_umatrix = self.att_u(inst_features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, inst_features)

        bag_probability = self.classifier(bag_features)

        return bag_probability, gated_attention


