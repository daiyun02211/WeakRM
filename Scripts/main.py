import argparse
from training import train_model
from testing import eval_model
from config import Config
from utils import create_folder, str2bool


parser = argparse.ArgumentParser(description="Train and evaluate WeakRM")
parser.add_argument('--training', default=True, type=str2bool, nargs='?',
                    help='training or evaluation')
parser.add_argument('--input_dir', default='./Data/m7G/processed/', type=str,
                    help='Path to processed data directory')
parser.add_argument('--model_name', default='WeakRM', type=str,
                    help='One of [WeakRM, WeakRMLSTM, WSCNN, WSCNNLSTM]')
parser.add_argument('--fusion_method', default='Max', type=str,
                    help='One of [MAX, AVG, NOISY]')
parser.add_argument('--epoch', default=20, type=int,
                    help='The number of epoch')
parser.add_argument('--lr_init', default=1e-4, type=float,
                    help='Initial learning rate')
parser.add_argument('--lr_decay', default=1e-5, type=float,
                    help='Decayed learning rate')
parser.add_argument('--len', default='50', type=int,
                    help='Instance length')
parser.add_argument('--stride', default='10', type=int,
                    help='Instance stride')
parser.add_argument('--cropping', default=False, type=str2bool, nargs='?',
                    help='Activate ramdon cropping')
parser.add_argument('--cp_dir', default='./Data/m7G/processed/cp_dir/',
                    type=str, help='Path to checkpoint directory')
parser.add_argument('--saving', default=False, type=str2bool, nargs='?',
                    help='Whether save weights during training')
parser.add_argument('--cp_name', default=None, type=str,
                    help='Name of saved checkpoint')

args = parser.parse_args()

training = args.training

c = Config
c.inst_len = args.len
c.inst_stride = args.stride
c.model_name = args.model_name
c.merging = args.fusion_method
c.data_dir = args.input_dir

c.epoch = args.epoch
c.lr_init = args.lr_init
c.lr_decay = args.lr_decay

cp_dir = args.cp_dir
create_folder(cp_dir)

if args.training & args.saving:
    if c.model_name.startswith('Weak'):
        c.cp_path = cp_dir + c.model_name + '.h5'
    else:
        c.cp_path = cp_dir + c.model_name + '_' + c.merging + '.h5'
elif args.training & (not args.saving):
    c.cp_path = None
elif (not args.training) & (not args.cp_name):
    if c.model_name.startswith('Weak'):
        c.cp_path = cp_dir + c.model_name + '.h5'
    else:
        c.cp_path = cp_dir + c.model_name + '_' + c.merging + '.h5'
elif (not args.training) & args.cp_name:
    c.cp_path = cp_dir + args.cp_name + '.h5'


if __name__ == '__main__':
    if training:
        train_model(c)
    else:
        eval_model(c)
