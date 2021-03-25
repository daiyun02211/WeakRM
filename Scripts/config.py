import numpy as np


class Config(object):
    inst_len = 50
    inst_stride = 10

    cropping = False
    nt_crop = 400
    crop_threshold = int(((nt_crop - inst_len) / inst_stride) + 1)

    epoch = 20
    lr_init = 1e-4
    lr_decay = 1e-5

    eval_after_train = True

    model_name = 'WeakRM'
    merging = 'NOISY'

    data_dir = '../data/m7G/'
    cp_path = data_dir + 'cp_dir/' + model_name + '.h5'

    threshold = 0.5


if __name__ == '__main__':
    c = Config()