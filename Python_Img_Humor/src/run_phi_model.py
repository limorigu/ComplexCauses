from torch.optim.lr_scheduler import StepLR
from utils import train_phi_models
from nets import CauseEffect
from data import get_img_sim_loaders_by_cov, \
    get_humicroedit_loaders_by_cov
from datetime import datetime
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


def train_phis(args, train=True, nested_model=False):
    """
   Wrapper function for the training of a phis estimation model
   for as many epochs as user specified. Training per epoch done by
   train_phi_models() in train_test_utils file, but all relevant inputs to
   train_phi_models are initialized here.

     Input:
     - args (run config)
     - train (flag to specify whether to use train or test split of data)
     - nested_model (flag to specify if to train full phi~Z,W model
     or phi~Z model for cond. ind. test)

     Output: model (trained phis pred. model) """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    if not nested_model:
        model = CauseEffect(args=args,
                            inputSize=args.inputSize_g,
                            outputSize=args.outputSize_g,
                            hidden_dim=args.hidden_dim_g).to(device)
    else:
        model = CauseEffect(args=args,
                            inputSize=args.inputSize_g - args.W_dim,
                            outputSize=args.outputSize_g,
                            hidden_dim=args.hidden_dim_g).to(device)
    print("model: ", model)

    if args.dataset == 'PertImgSim':
        loader_fn = get_img_sim_loaders_by_cov
    elif args.dataset == 'Humicroedit':
        loader_fn = get_humicroedit_loaders_by_cov
    else:
        raise NotImplementedError

    if train:
        train_loader = loader_fn(args=args, split='train_seen_train')
        lr = args.lr1
        epochs = args.epochs1
    else:
        train_loader = loader_fn(args=args, split='test_unseen_train')
        lr = args.lr2
        epochs = args.epochs2
    # initialize optimizer
    if args.optimizer == 'sgd':
        optimizer_fn = optim.SGD
        optimizer = optimizer_fn(model.parameters(), lr=lr,
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer_fn = optim.Adam
        optimizer = optimizer_fn(model.parameters(), lr=lr,
                                 weight_decay=args.weight_decay,
                                 betas=(args.beta1, args.beta2))
    else:
        raise NotImplementedError
    loss_criterion = F.mse_loss

    # optional: activate optimizer lr scheduler
    scheduler = StepLR(optimizer, step_size=args.schd_step_size, gamma=args.gamma)

    # define tensorboard logs saving path
    tensor_log_name = '/' + args.dataset + \
                      '/train_batch_size_' + str(args.train_batch_size) + \
                      '/num_batches_train_' + str(len(train_loader)) + \
                      '/hidden_dim' + str(args.hidden_dim_g) + \
                      '/opt_' + args.optimizer + '/lr_' + str(lr) + \
                      '/epochs_' + str(epochs) + \
                      '/train_' + str(train) + '/' + \
                      '/dropout_' + str(args.dropout) + \
                      datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # initialize tesnorboard writer
    writer = SummaryWriter(
        args.tensorboard_dir + tensor_log_name)

    # inner_loss = 0
    for epoch in range(1, epochs + 1):
        print("epoch " + str(epoch) + ": " + str(datetime.now()))
        losses, ex_vars = train_phi_models(args=args, model=model, loss_criterion=loss_criterion, 
                                            optimizer=optimizer, scheduler=scheduler, 
                                            train_loader=train_loader, nested_model=nested_model)

        writer.add_scalar('Train_loss_/',
                          np.asarray(losses).mean(), epoch)
        writer.add_scalar('Train_ex_var_/',
                          np.asarray(ex_vars).mean(), epoch)
    writer.flush()
    writer.close()
    print("end of train/test", datetime.now())
    return model
