from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from torch.utils.data import DataLoader
from torch import optim
import random
import pandas as pd
from data import get_full_vector_img_sim, get_img_sim_loaders_by_cov, \
    get_full_vector_humicroedit, DataIter, get_humicroedit_loaders_by_cov
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from nets import MLP
from run_phi_model import train_phis
from .train_test_utils import test_y_model, train, train_y_model
import torch.nn.functional as F

sns.set_style("whitegrid")


def evaluate_models(args, model_y,
                    baseline=False, treatment='W',
                    MLP=False, model_phis=None,
                    pass_model_phis=False):
    """ helper function to evaluate given model,
    based on Y predictions and Y ground truth.

     Input:
     - args (run config from user)
     - model_y (lasso y pred. model to evaluate),
     - baseline (flag indicating if baseline or our model)
     - treatment (flag defining intervention variable)
     - MLP (flag indicating if Y model is MLP or not)
     Output:
    - (optional: model_weights, only for our method),
    - test_real_Y (optional: only needed from one call,
                    remains the same throughout),
    - test_est_Y,
    - test_diff_Y,
    - test_rltv_diff_Y
    - test_sqr_diff_Y """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")
    print("device: ", device)

    test_real_Y = []
    test_est_Y = []
    test_diff_Y = []
    test_rltv_diff_Y = []
    test_sqr_diff_Y = []

    if args.dataset == 'PertImgSim':
        Z, W, X, Y, img_ind = get_full_vector_img_sim(args, split='test_unseen_test')
    elif args.dataset == 'Humicroedit':
        Z, W, X, Y, phis = get_full_vector_humicroedit(args, split='test_unseen_test')
    else:
        raise NotImplementedError

    if not baseline:
        if model_phis is None:
            model_phis = train_phis(args, train=False)
        if args.Y_target == 'continuous':
            Y_hat, model_weights = test_y_model(args, model_phis, model_y)
            Y = Y.detach().cpu().numpy().squeeze(axis=1)
        else:
            raise NotImplementedError
    else:
        Y = Y.detach().cpu().numpy().squeeze(axis=1)
        if treatment == 'W':
            if args.Y_target == 'continuous':
                features_test_model_Y = torch.cat((Z, W), axis=1)
                if not MLP:
                    Y_hat = \
                        model_y.predict(features_test_model_Y.cpu().detach().numpy())
                else:
                    print("model_y: ", model_y)
                    Y_hat = \
                        model_y(features_test_model_Y.to(device)) \
                            .cpu().detach().numpy()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    if args.Y_target == "continuous":
        test_real_Y.append(Y)
        test_est_Y.append(Y_hat)
        test_diff_Y.append(abs(Y_hat - Y))
        test_rltv_diff_Y.append(abs((Y - Y_hat) / Y))
        test_sqr_diff_Y.append((Y_hat - Y) ** 2)

        if baseline:
            return np.concatenate(test_real_Y), \
                   np.concatenate(test_est_Y), \
                   np.concatenate(test_diff_Y), \
                   np.concatenate(test_rltv_diff_Y), \
                   np.concatenate(test_sqr_diff_Y)
        else:
            if not pass_model_phis:
                return model_weights, \
                       np.concatenate(test_real_Y), \
                       np.concatenate(test_est_Y), \
                       np.concatenate(test_diff_Y), \
                       np.concatenate(test_rltv_diff_Y), \
                       np.concatenate(test_sqr_diff_Y)
            else:
                return model_weights, \
                       np.concatenate(test_real_Y), \
                       np.concatenate(test_est_Y), \
                       np.concatenate(test_diff_Y), \
                       np.concatenate(test_rltv_diff_Y), \
                       np.concatenate(test_sqr_diff_Y), \
                       model_phis
    else:
        raise NotImplementedError


def get_baseline_model_and_data_load(args, baseline_type,
                                     treatment, train_flag=True):
    """
   Function for the training of a baseline model.
   User specified argument (in args) determins the type
   of model trained. Currently supports BART (from bartpy)
   and Random Forest (from sklearn).

     Input:
     - args (run config)
     - baseline_type
     - treatment (definition of intervention variable)
     - train_flag (flag indicating if train or test)
     Output: model (trained baseline model) """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'PertImgSim':
        Z_train, W_train, X_train, Y_train, phis_train = \
            get_full_vector_img_sim(args, split='train_seen_train')
    elif args.dataset == 'Humicroedit':
        Z_train, W_train, X_train, Y_train, phis_train = \
            get_full_vector_humicroedit(args, split='train_seen_train')
    else:
        raise NotImplementedError

    data = Z_train, W_train, X_train, Y_train, phis_train

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    model = \
        get_baseline_model(args, data, device,
                           baseline_type=baseline_type,
                           treatment=treatment,
                           train_flag=train_flag)

    del data, Z_train, W_train, X_train, Y_train, phis_train

    return model


def get_baseline_model(args, data, device,
                       baseline_type='LassoReg', treatment='X',
                       train_flag=True):
    """
   Function for the training of a baseline model.
   User specified argument (in args) determins the type
   of model trained. Currently supports BART (from bartpy)
   and Random Forest (from sklearn).

     Input:
     - args (run config)
     - data (data to be used for models)
     - device (device to be used for Pytorch)
     - baseline_type (flag specifying baseline type)
     - treamtnet (flag spacifying if intervention on X or W)
     - train_flag (flag to specify if train)
     Output: model (trained baseline model) """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Z_train, W_train, X_train, Y_train, phis_train = data

    if treatment == 'X':
        if baseline_type == 'LassoReg':
            lasso = linear_model.Lasso()
            predictor = GridSearchCV(lasso,
                                     {'alpha': list(map(float, args.alphas))})
            features_train = np.concatenate((Z_train, X_train), axis=1)
            predictor.fit(features_train,
                          Y_train.squeeze())
            model = predictor.best_estimator_
        elif baseline_type == 'SVR':
            model = SVR(kernel='rbf')
            features_train = np.concatenate((Z_train, X_train), axis=1)
            model.fit(features_train,
                      Y_train.squeeze())

        elif baseline_type == 'RF':
            model = RandomForestRegressor(min_samples_splitint=5)
            features_train = np.concatenate((Z_train, X_train), axis=1)
            model.fit(features_train,
                      Y_train.squeeze())

        elif baseline_type == 'GB':
            model = GradientBoostingRegressor(max_depth=5)
            features_train = np.concatenate((Z_train, X_train), axis=1)
            model.fit(features_train,
                      Y_train.squeeze())

        elif baseline_type == 'MLP':
            if train_flag:
                lr = args.lr1
                epochs = args.epochs1
            else:
                lr = args.lr2
                epochs = args.epochs2
            model = MLP(args,
                        args.X_dim + args.Z_dim,
                        args.hidden_dim_g,
                        args.Y_dim).to(device)

            Z_train, X_train, Y_train = \
                Z_train.to(device), X_train.to(device), Y_train.to(device)
            features_train = torch.cat((Z_train, X_train), axis=1)
            train_loader = DataLoader(DataIter(features_train, Y_train),
                                      batch_size=args.test_batch_size)

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
            for _ in range(1, epochs + 1):
                train(model, device, optimizer, train_loader, loss_criterion, scheduler=None)

        else:
            raise NotImplementedError

    elif treatment == 'W':
        if baseline_type == 'LassoReg':
            lasso = linear_model.Lasso()
            predictor = GridSearchCV(lasso,
                                     {'alpha': list(map(float, args.alphas))})

            features_train_model_Y = np.concatenate((Z_train, W_train), axis=1)
            predictor.fit(features_train_model_Y,
                          Y_train.squeeze())  # Fit the model
            model = predictor.best_estimator_

        elif baseline_type == 'SVR':
            model = SVR(kernel='rbf')
            features_train = np.concatenate((Z_train, W_train), axis=1)
            model.fit(features_train,
                      Y_train.squeeze())  # Fit the model

        elif baseline_type == 'RF':
            model = RandomForestRegressor(min_samples_split=5)
            features_train = np.concatenate((Z_train, W_train), axis=1)
            model.fit(features_train,
                      Y_train.squeeze())  # Fit the model

        elif baseline_type == 'GB':
            model = GradientBoostingRegressor(max_depth=5)
            features_train = np.concatenate((Z_train, W_train), axis=1)
            model.fit(features_train,
                      Y_train.squeeze())  # Fit the model

        elif baseline_type == 'MLP':
            if train_flag:
                lr = args.lr1
                epochs = args.epochs1
            else:
                lr = args.lr2
                epochs = args.epochs2

            model = MLP(args,
                        args.W_dim + args.Z_dim,
                        args.hidden_dim_g,
                        args.Y_dim).to(device)

            Z_train, W_train, Y_train = \
                Z_train.to(device), W_train.to(device), Y_train.to(device)
            features_train = torch.cat((Z_train, W_train), axis=1)
            train_loader = DataLoader(DataIter(features_train, Y_train),
                                      batch_size=args.test_batch_size)

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
            for _ in range(1, epochs + 1):
                train(model, device, optimizer, train_loader, loss_criterion, scheduler=None)
        else:
            raise NotImplementedError
    elif treatment == 'phis':
        lasso = linear_model.Lasso()
        predictor = GridSearchCV(lasso,
                                 {'alpha': list(map(float, args.alphas))})

        features_train_model_Y = phis_train
        predictor.fit(features_train_model_Y,
                      Y_train.squeeze())  # Fit the model
        model = predictor.best_estimator_
    return model


def MSE(Y, Y_hat):
    """
   Helper function to compute mean squared error

     Input:
     - Y (traget MSE computed against)
     - Y_hat (prediction MSE computed w.r.t)
     Output: MSE """
    return np.mean((Y_hat - Y) ** 2)


def Y_pred_as_perc_labels(args, test_est_Y,
                          num_perc_of_labels,
                          plot=True, normalize_MSE=False):
    """
   Code to compute and plot figure 4 in the paper: MSE performance
   on Y prediction as a function of % of Y labels available for training.
   Our method requires no Y labels, and therefore remains constant.
   Compared to baselines to compute E[Y|do(w), Z] with linear, SVM,
   Random Forest and Gradient Boosting models.

     Input:
     - args (run config)
     - test_est_Y (our model Y estimation)
     - num_perc_of_labels (how many splits of
     training data to show % performance over) """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    if args.dataset == 'PertImgSim':
        Z_train, W_train, X_train, Y_train, _ = \
            get_full_vector_img_sim(args, split='test_unseen_train')

        Z_test, W_test, X_test, Y_test, _ = \
            get_full_vector_img_sim(args, split='test_unseen_test')
    elif (args.dataset == 'Humicroedit'):
        Z_train, W_train, X_train, Y_train, _ = \
            get_full_vector_humicroedit(args, split='test_unseen_train')

        Z_test, W_test, X_test, Y_test, _ = \
            get_full_vector_humicroedit(args, split='test_unseen_test')
    else:
        raise NotImplementedError

    Y_test = Y_test.detach().cpu().numpy().squeeze()

    if not normalize_MSE:
        our_method = [MSE(Y_test, test_est_Y)] * num_perc_of_labels
    else:
        normalize_factor = (np.max(Y_test) - np.min(Y_test)) ** 2
        our_method = [MSE(Y_test, test_est_Y) / normalize_factor] * num_perc_of_labels
    linear_Y_Z_W = []
    nonlinear_Y_Z_W = []
    nonlinear_RF_Y_Z_W = []
    nonlinear_GB_Y_Z_W = []

    split = Z_train.shape[0] / num_perc_of_labels
    for trial in range(args.num_trials_exp):
        data_temp = \
            torch.cat((Z_train, W_train, X_train, Y_train), axis=1)
        random.shuffle(data_temp)
        Z_train = data_temp[:, :Z_train.shape[1]]

        W_train = data_temp[:, Z_train.shape[1]
                               :Z_train.shape[1] +
                                W_train.shape[1]]

        X_train = data_temp[:, Z_train.shape[1] +
                               W_train.shape[1]:
                               Z_train.shape[1] +
                               W_train.shape[1] +
                               X_train.shape[1]]

        Y_train = data_temp[:, Z_train.shape[1] +
                               W_train.shape[1] +
                               X_train.shape[1]:]

        print()
        linear_Y_Z_W_i = []
        nonlinear_Y_Z_W_i = []
        nonlinear_RF_Y_Z_W_i = []
        nonlinear_GB_Y_Z_W_i = []

        for i in range(1, num_perc_of_labels + 1):
            data = Z_train[:int(split * i), :], W_train[:int(split * i), :], \
                   X_train[:int(split * i), :], Y_train[:int(split * i), :], _

            print("######################################")
            print("baseline linear Y|Z,W model split " + str(int(split * i)))
            print("######################################")
            baseline_model_Y = get_baseline_model(args, data, device,
                                                  baseline_type='LassoReg', treatment='W')
            features_test_model_Y = torch.cat((Z_test, W_test), axis=1)
            Y_hat = \
                baseline_model_Y.predict(features_test_model_Y.detach().cpu().numpy())

            if not normalize_MSE:
                linear_Y_Z_W_i.append(MSE(Y_test, Y_hat))
            else:
                linear_Y_Z_W_i.append(MSE(Y_test, Y_hat) /
                                      normalize_factor)
            assert Y_test.shape == Y_hat.shape

            print("######################################")
            print("baseline nonlinear SVR Y|Z,W model split " + str(int(split * i)))
            print("######################################")
            baseline_model_Y_nonlinear = get_baseline_model(args, data, device,
                                                            baseline_type='SVR', treatment='W')
            features_test_model_Y = torch.cat((Z_test, W_test), axis=1)
            Y_hat_nonlinear = \
                baseline_model_Y_nonlinear.predict(
                    features_test_model_Y.detach().cpu().numpy())

            if not normalize_MSE:
                nonlinear_Y_Z_W_i.append(MSE(Y_test, Y_hat_nonlinear))
            else:
                nonlinear_Y_Z_W_i.append(MSE(Y_test, Y_hat_nonlinear) /
                                         normalize_factor)
            assert Y_test.shape == Y_hat_nonlinear.shape

            print("######################################")
            print("baseline nonlinear RF Y|Z,W model split " + str(int(split * i)))
            print("######################################")
            baseline_model_Y_nonlinear_RF = get_baseline_model(args, data, device,
                                                               baseline_type='RF', treatment='W')
            features_test_model_Y = torch.cat((Z_test, W_test), axis=1)
            Y_hat_nonlinear_RF = \
                baseline_model_Y_nonlinear_RF.predict(
                    features_test_model_Y.detach().cpu().numpy())

            if not normalize_MSE:
                nonlinear_RF_Y_Z_W_i.append(MSE(Y_test, Y_hat_nonlinear_RF))
            else:
                nonlinear_RF_Y_Z_W_i.append(MSE(Y_test, Y_hat_nonlinear_RF) /
                                            normalize_factor)
            assert Y_test.shape == Y_hat_nonlinear_RF.shape

            print("######################################")
            print("baseline nonlinear GB Y|Z,W model split " + str(int(split * i)))
            print("######################################")
            baseline_model_Y_nonlinear_GB = get_baseline_model(args, data, device,
                                                               baseline_type='GB', treatment='W')
            features_test_model_Y = torch.cat((Z_test, W_test), axis=1)
            Y_hat_nonlinear_GB = \
                baseline_model_Y_nonlinear_GB.predict(
                    features_test_model_Y.detach().cpu().numpy())

            if not normalize_MSE:
                nonlinear_GB_Y_Z_W_i.append(MSE(Y_test, Y_hat_nonlinear_GB))
            else:
                nonlinear_GB_Y_Z_W_i.append(MSE(Y_test, Y_hat_nonlinear_GB) /
                                            normalize_factor)
            assert Y_test.shape == Y_hat_nonlinear_GB.shape

        linear_Y_Z_W.append(linear_Y_Z_W_i)
        nonlinear_Y_Z_W.append(nonlinear_Y_Z_W_i)
        nonlinear_RF_Y_Z_W.append(nonlinear_RF_Y_Z_W_i)
        nonlinear_GB_Y_Z_W.append(nonlinear_GB_Y_Z_W_i)

    # mean over trials, output will be of size num_perc_of_labels
    linear_Y_Z_W = np.mean(np.array(linear_Y_Z_W), axis=0)
    nonlinear_Y_Z_W = np.mean(np.array(nonlinear_Y_Z_W), axis=0)
    nonlinear_RF_Y_Z_W = np.mean(np.array(nonlinear_RF_Y_Z_W), axis=0)
    nonlinear_GB_Y_Z_W = np.mean(np.array(nonlinear_GB_Y_Z_W), axis=0)

    if plot:
        BIGGER_SIZE = 16
        plt.clf()
        plt.rc('font', size=BIGGER_SIZE)
        plt.rc('axes', labelsize=BIGGER_SIZE)
        plt.rc('xtick', labelsize=BIGGER_SIZE)
        plt.rc('ytick', labelsize=BIGGER_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)
        plt.plot(list(range(len(linear_Y_Z_W))), linear_Y_Z_W,
                 label='linear', alpha=0.4, linewidth=3.5)
        plt.plot(list(range(len(nonlinear_Y_Z_W))), nonlinear_Y_Z_W,
                 label='SVR', alpha=0.4, linewidth=3.5)
        plt.plot(list(range(len(nonlinear_RF_Y_Z_W))), nonlinear_RF_Y_Z_W,
                 label='RF', alpha=0.4, linewidth=3.5)
        plt.plot(list(range(len(nonlinear_GB_Y_Z_W))), nonlinear_GB_Y_Z_W,
                 label='GB', alpha=0.4, linewidth=3.5)
        plt.plot(list(range(len(our_method))), our_method,
                 label=r'$\bf{ours}$', alpha=0.4, linewidth=3.5, color='red')

        def larger_axlim(axlim):
            """ argument axlim expects 2-tuple
                returns slightly larger 2-tuple """
            axmin, axmax = axlim
            axrng = axmax - axmin
            new_min = axmin + 0.1 * axrng
            new_max = axmax - 0.1 * axrng
            return new_min, new_max

        plt.xlim(larger_axlim(plt.xlim()))
        plt.xticks(list(range(len(linear_Y_Z_W))),
                   [i * 10 for i in range(1, len(linear_Y_Z_W) + 1)])
        plt.xlabel("% labeled")
        plt.ylabel("MSE")
        plt.legend()

        Path(args.save_loc).mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_loc + '/MSE_vs_Y_labels_perc.png',
                    bbox_inches='tight')
    else:
        return linear_Y_Z_W, nonlinear_Y_Z_W, \
               nonlinear_RF_Y_Z_W, nonlinear_GB_Y_Z_W, \
               our_method


def save_Y_coeffs(args, fttd_model_params):
    """
   Code to compute and plot parameters of
   fitted y prediction model against ground truth

     Input:
     - args (run config)
     - fttd_model_params (fitted parameters of our y model) """
    fttd_model_params = fttd_model_params[0]
    read_in = np.load(args.data_save_dir + '_params.npz', allow_pickle=True)
    idx = read_in.files[0]
    real_params = read_in[idx]

    df_params = \
        pd.DataFrame.from_dict({
            'real_params': abs(real_params),
            'fttd_params': abs(fttd_model_params)})

    df_params.to_csv(args.save_loc + "/df_params_" +
                     args.dataset + ".csv")

    print("real_params: ", real_params)
    print("fttd_model_params: ", fttd_model_params)


def test_model_phis_residuals(args, model,
                              nested_model=False):
    """
   Code to produce phi model predictions from the models
   phis ~ Z,W and phis ~ Z, for conditional independence test.

     Input:
     - args (run config)
     - model (model to get predictions from)
     - nested model (flag indicating if nested
     (phis ~ Z) or extended (phis ~ Z,W) model """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    if args.dataset == 'PertImgSim':
        loader_fn = get_img_sim_loaders_by_cov
        num_phis = args.Z_dim // (args.window_size_phi ** 2)
    elif args.dataset == 'Humicroedit':
        loader_fn = get_humicroedit_loaders_by_cov
        num_phis = args.num_phis
    else:
        raise NotImplementedError

    test_loader = loader_fn(args=args, split='train_seen_test')

    residuals = {i: [] for i in range(num_phis)}
    for batch_idx, (Z, W, X, Y, phis, idx) in enumerate(test_loader):
        Z, W, X, Y, phis = \
            Z.to(device), W.to(device), X.to(device), \
            Y.to(device), phis.to(device)

        if not nested_model:
            combined_input = torch.cat((Z, W), axis=1)
            phi_hats = model(combined_input)
        else:
            phi_hats = model(Z)

        for i in range(num_phis):
            real_phi_i = phis[:, i].unsqueeze(dim=1)
            pred_phi_i = phi_hats[i]
            residuals[i] += list((real_phi_i - pred_phi_i)
                                 .detach().cpu().numpy().reshape(-1))

    residuals_mat = np.array(list(residuals.values()))
    assert residuals_mat.shape == \
           (num_phis, args.dataset_length_test_seen)
    residuals_mat = residuals_mat.T
    Path(args.save_loc).mkdir(parents=True, exist_ok=True)
    np.savetxt(args.save_loc + '/residuals_nested=' +
               str(nested_model) + '_' + args.dataset + '.csv',
               residuals_mat, delimiter=",")


def plot_MSE_results(args, means_diff_Y, error_bars):
    """
    Plot MSE results, reproducting Fig. 6, including a sensitivity
    analysis on parameters in the structural equation for Y. Thus,
    we take in as input the means and error bars from runs for N problems,
    where N = args.Y_coeffs_config_num

     Input:
     - args (run config)
     - means_diff_Y (means of runs over N problems)
     - error_bars (error bars of runs over N problems) """
    BIGGER_SIZE = 16
    plt.clf()
    plt.rc('font', size=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE)
    plt.rc('ytick', labelsize=BIGGER_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    fig, ax = plt.subplots()

    plt.plot(list(range(len(means_diff_Y['linear_Y_Z_W']))), means_diff_Y['linear_Y_Z_W'],
             label='linear', alpha=0.4, linewidth=3.5)
    ax.fill_between(list(range(len(means_diff_Y['linear_Y_Z_W']))),
                    means_diff_Y['linear_Y_Z_W'] - error_bars['linear_Y_Z_W'],
                    means_diff_Y['linear_Y_Z_W'] + error_bars['linear_Y_Z_W'],
                    alpha=.1)

    plt.plot(list(range(len(means_diff_Y['nonlinear_Y_Z_W']))), means_diff_Y['nonlinear_Y_Z_W'],
             label='SVR', alpha=0.4, linewidth=3.5)
    ax.fill_between(list(range(len(means_diff_Y['nonlinear_Y_Z_W']))),
                    means_diff_Y['nonlinear_Y_Z_W'] - error_bars['nonlinear_Y_Z_W'],
                    means_diff_Y['nonlinear_Y_Z_W'] + error_bars['nonlinear_Y_Z_W'],
                    alpha=.1)

    plt.plot(list(range(len(means_diff_Y['nonlinear_RF_Y_Z_W']))), means_diff_Y['nonlinear_RF_Y_Z_W'],
             label='RF', alpha=0.4, linewidth=3.5)
    ax.fill_between(list(range(len(means_diff_Y['nonlinear_RF_Y_Z_W']))),
                    means_diff_Y['nonlinear_RF_Y_Z_W'] - error_bars['nonlinear_RF_Y_Z_W'],
                    means_diff_Y['nonlinear_RF_Y_Z_W'] + error_bars['nonlinear_RF_Y_Z_W'],
                    alpha=.1)

    plt.plot(list(range(len(means_diff_Y['nonlinear_GB_Y_Z_W']))), means_diff_Y['nonlinear_GB_Y_Z_W'],
             label='GB', alpha=0.4, linewidth=3.5)
    ax.fill_between(list(range(len(means_diff_Y['nonlinear_GB_Y_Z_W']))),
                    means_diff_Y['nonlinear_GB_Y_Z_W'] - error_bars['nonlinear_GB_Y_Z_W'],
                    means_diff_Y['nonlinear_GB_Y_Z_W'] + error_bars['nonlinear_GB_Y_Z_W'],
                    alpha=.1)

    plt.plot(list(range(len(means_diff_Y['our_method']))), means_diff_Y['our_method'],
             label=r'$\bf{ours}$', alpha=0.4, linewidth=3.5, color='red')
    ax.fill_between(list(range(len(means_diff_Y['our_method']))),
                    means_diff_Y['our_method'] - error_bars['our_method'],
                    means_diff_Y['our_method'] + error_bars['our_method'],
                    alpha=.1)

    def larger_axlim(axlim):
        """ argument axlim expects 2-tuple
            returns slightly larger 2-tuple """
        axmin, axmax = axlim
        axrng = axmax - axmin
        new_min = axmin + 0.1 * axrng
        new_max = axmax - 0.1 * axrng
        return new_min, new_max

    plt.xlim(larger_axlim(plt.xlim()))
    plt.xticks(list(range(len(means_diff_Y['linear_Y_Z_W']))),
               [i * 10 for i in range(1, len(means_diff_Y['linear_Y_Z_W']) + 1)])
    plt.xlabel("% labeled")
    plt.ylabel("MSE")
    plt.legend()

    Path(args.save_loc).mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_loc + \
                '/unif_diffY{}_MSE_vs_Y_labels_perc.png'.format(args.Y_coeffs_config_num),
                bbox_inches='tight')
