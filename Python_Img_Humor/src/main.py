import argparse
from datetime import datetime

from run_results_and_baselines \
    import get_models, get_results_viz_baselines,\
    Y_coeffs_sensitivity
from utils.run_utils import fill_config_args
import os
import torch
import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Run Experiment.')

# Setup config
parser.add_argument('--config-causal-effect', type=str,
                    default='configs/causal_effect_config.yml',
                    dest='config_causal_effect',
                    help='config file')

# Data loading etc.
parser.add_argument('--dataset', '-d', dest='dataset', type=str,
                    help='Dataset')
parser.add_argument('--dataset-length', dest='dataset_length', type=int,
                    help='Dataset length')
parser.add_argument('--dataset-length-test-seen', dest='dataset_length_test_seen', type=int,
                    help='Dataset length test seen')
parser.add_argument('--dataset-length-test', dest='dataset_length_test', type=int,
                    help='Dataset length test')
parser.add_argument('--data-save-dir', dest='data_save_dir', type=str,
                    help='Dataset save directory')
parser.add_argument('--Z-dim', type=int,
                    dest='Z_dim', help='W dimensions')
parser.add_argument('--W-dim', type=int,
                    dest='W_dim', help='W dimensions')
parser.add_argument('--X-dim', type=int,
                    dest='X_dim', help='X dimensions')
parser.add_argument('--Y-dim', type=int,
                    dest='Y-dim', help='Y dimensions')
parser.add_argument('--lambdas', dest='lambda',
                    help='lambda values to try for obj. func.')
parser.add_argument('--tensorboard-dir', dest='tensorboard_dir',
                    type=str, help='tensorboard dir')
parser.add_argument('--train-test-split', dest='train_test_split',
                    type=bool, help='train test split')

# Optimization
parser.add_argument('--ablation', dest='ablation',
                    type=str, help='ablation')
parser.add_argument('--treatment', '-t', dest='treatment', type=str,
                    help='Treatment node')
parser.add_argument('--optimizer', '-o', dest='optimizer', type=str,
                    help='optimizer')
parser.add_argument('--lr1', type=float, dest='lr1',
                    help='learning rate model 1')
parser.add_argument('--lr2', type=float, dest='lr2',
                    help='learning rate model 2')
parser.add_argument('--dropout', type=float, dest='dropout',
                    help='dropout for MLP')
parser.add_argument('--schd-step-size', type=float, dest='schd_step_size',
                    help='schd step size for sched')
parser.add_argument('--gamma', type=float, dest='gamma',
                    help='gamma for sched')
parser.add_argument('--weight-decay', type=float, dest='weight_decay',
                    help='weight decay')
parser.add_argument('--momentum', type=float, dest='momentum',
                    help='momentum')
parser.add_argument('--beta1', type=float, dest='beta1',
                    help='beta1')
parser.add_argument('--beta2', type=float, dest='beta2',
                    help='beta2')
parser.add_argument('--epochs1', '-e', type=int, dest='epochs1',
                    help='Number of epochs 1')
parser.add_argument('--epochs2', type=int, dest='epochs2',
                    help='Number of epochs 2')
parser.add_argument('--update-inner-stepsize', type=int, dest='update_inner_stepsize',
                    help='update inner stepsize')
parser.add_argument('--Y-target', dest='Y_target',
                    type=str, help='Y target')
parser.add_argument('--perc-labels-for-Y-pred', dest='perc_labels_for_Y_pred',
                    type=int, help='perc labels for Y pred')
parser.add_argument('--num-trials-exp1', dest='num_trials_exp1',
                    type=int, help='num trials for exp1')

# dims for nets
parser.add_argument('--inputSize-g', type=int, dest='inputSize_g',
                    help='input size g')
parser.add_argument('--outputSize-g', type=int, dest='outputSize_g',
                    help='output size g')
parser.add_argument('--hidden-dim-g', type=int, dest='hidden_dim_g',
                    help='hiddendim g')
parser.add_argument('--g-parametrize', dest='g_parametrize',
                    type=str, help='g parametrize')
parser.add_argument('--g-optimize', dest='g_optimize',
                    type=str, help='g optimize')
# batching
parser.add_argument('--train-batch-size', '-b', type=int,
                    dest='train_batch_size',
                    help='train batch size')
parser.add_argument('--valid-batch-size', type=int,
                    dest='valid_batch_size',
                    help='valid batch size')
parser.add_argument('--test-batch-size', type=int,
                    dest='test_batch_size',
                    help='test batch size')

# Sim_fit
parser.add_argument('--baseline-type', dest='baseline_type',
                    type=str, help='baseline type')
parser.add_argument('--window-size-phi', type=int, dest='window_size_phi',
                    help='window size phi')
parser.add_argument('--num-phis', type=int, dest='num_phis',
                    help='num phis')

# Utils
parser.add_argument("--no-cuda", action="store_true", dest="no_cuda",
                    help="Activate stable rank normalization")
parser.add_argument('--seed', type=int,
                    dest='seed', help='Seed')

# Visualize
parser.add_argument('--save-loc', dest='save_loc',
                    type=str, help='visualization save location')
parser.add_argument('--save-loc-base', dest='save_loc_base',
                    type=str, help='general save location')
parser.add_argument('--save-name', dest='save_name',
                    type=str, help='visualization save name')


# run methods
###############################################
# Optimize model, evaluate and visualize
###############################################
def main():
    start = datetime.now()
    print("begin script: " + str(start))
    args = parser.parse_args()
    args = fill_config_args(args)
    print("model train")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_phis_seen, model_y = \
        get_models(args)
    print("evaluate")
    model_phis_unseen = \
        get_results_viz_baselines(args, model_y)

    # Note: this function reproduces figure 8 in the masnucript. It takes longer to run then the rest!
    Y_coeffs_sensitivity(args, model_phis_seen,
                         model_phis_unseen)
    end = datetime.now()
    print("end script: " + str(end))
    print("total runtime: " + str(end-start))


if __name__ == '__main__':
    main()
