from utils import evaluate_models, visualize_ATEs, \
    train_y_model, Y_pred_as_perc_labels, save_Y_coeffs, \
    test_model_phis_residuals, plot_MSE_results
from run_phi_model \
    import train_phis
import numpy as np
import pandas as pd
from pathlib import Path


def get_models(args):
    """ Training routine for phi and y model training.
     Input:
     - args (run config)
    Output:
    - models_phis (trained model for phi prediction)
    - model_y (trained Lasso model for Y prediction) """
    ################################
    # train phi and y models
    ################################
    print("full model train")
    models_phis = \
        train_phis(args, train=True)
    models_phis_nested = \
        train_phis(args, train=True, nested_model=True)

    # compute and save residuals to file, to be used later in cond. ind. test
    test_model_phis_residuals(args, models_phis, nested_model=False)
    test_model_phis_residuals(args, models_phis_nested, nested_model=True)

    model_y = \
        train_y_model(args, models_phis)

    return models_phis, model_y


def get_results_viz_baselines(args, model_y):
    """ Evaluation routine for phi and y models.
    Will ultimately create visualizations of Y predictions
    via scatter plots, summary of results in a dataframe.

     Input:
     - args (run config)
    - model_y (trained model_y) """
    print("######################################")
    print("estimation model evaluate")
    print("######################################")
    if args.Y_target == 'continuous':
        model_weights, \
        test_real_Y, \
        test_est_Y, \
        test_diff_Y, \
        test_rltv_diff_Y, \
        test_sqrd_diff_Y, \
        model_phis_unseen = \
            evaluate_models(args, model_y,
                            pass_model_phis=True)
    else:
        raise NotImplementedError

    #########################################################
    # Visualize results: our estimation
    #########################################################
    # our estimation
    visualize_ATEs(Xs=test_est_Y,
                   Ys=test_real_Y,
                   x_name="est Y",
                   y_name="real Y",
                   save_loc=args.save_loc,
                   save_name="Y_est_vs_real")

    Y_pred_as_perc_labels(args, test_est_Y,
                          num_perc_of_labels=args.perc_labels_for_Y_pred)

    save_Y_coeffs(args, model_weights)

    return model_phis_unseen


def Y_coeffs_sensitivity(args, models_phis_seen,
                         models_phis_unseen):
    """
    Test the method for variation in coeffs in the sturcutral equation
    for Y. Average results over n different configurations of those parameters.
   Compared to baselines to compute E[Y|do(w), Z] with linear, SVM,
   Random Forest and Gradient Boosting models. Since our sensitivity analysis
   is over parameters in the structural equation for Y,
   we can use the already saved phis in the unseen W split from previous steps,
   as this won't vary. Thus, we take in that trained g model as input.

     Input:
     - args (run config)
     - models_phis_seen (trained model for phi prediction (in seen W split))
     - models_phis_unseen (trained model for phi prediction (in unseen W split)) """

    linear_Y_Z_W_diff_Ys = []
    nonlinear_Y_Z_W_diff_Ys = []
    nonlinear_RF_Y_Z_W_diff_Ys = []
    nonlinear_GB_Y_Z_W_diff_Ys = []
    our_method_diff_Ys = []

    for Y_coeffs_config_i in range(args.Y_coeffs_config_num):
        args.data_save_dir = args.Ydiff_data_save_dir + \
                             "{}".format(Y_coeffs_config_i)

        model_y = \
            train_y_model(args, models_phis_seen)

        print("######################################")
        print("estimation model evaluate")
        print("######################################")
        if args.Y_target == 'continuous':
            model_weights, \
            test_real_Y, \
            test_est_Y, \
            test_diff_Y, \
            test_rltv_diff_Y, \
            test_sqrd_diff_Y = \
                evaluate_models(args, model_y,
                                model_phis=models_phis_unseen)

        else:
            raise NotImplementedError

        #########################################################
        # Compare baselines and visualize results: our estimation
        #########################################################
        linear_Y_Z_W, nonlinear_Y_Z_W, \
        nonlinear_RF_Y_Z_W, nonlinear_GB_Y_Z_W, \
        our_method = \
            Y_pred_as_perc_labels(args, test_est_Y,
                                  num_perc_of_labels=args.perc_labels_for_Y_pred,
                                  plot=False, normalize_MSE=False)

        assert len(linear_Y_Z_W) == len(our_method)
        linear_Y_Z_W_diff_Ys.append(linear_Y_Z_W)
        nonlinear_Y_Z_W_diff_Ys.append(nonlinear_Y_Z_W)
        nonlinear_RF_Y_Z_W_diff_Ys.append(nonlinear_RF_Y_Z_W)
        nonlinear_GB_Y_Z_W_diff_Ys.append(nonlinear_GB_Y_Z_W)
        our_method_diff_Ys.append(our_method)

    assert np.array(linear_Y_Z_W_diff_Ys).shape == np.array(our_method_diff_Ys).shape
    linear_Y_Z_W_mean = np.mean(np.array(linear_Y_Z_W_diff_Ys), axis=0)
    nonlinear_Y_Z_W_mean = np.mean(np.array(nonlinear_Y_Z_W_diff_Ys), axis=0)
    nonlinear_RF_Y_Z_W_mean = np.mean(np.array(nonlinear_RF_Y_Z_W_diff_Ys), axis=0)
    nonlinear_GB_Y_Z_W_mean = np.mean(np.array(nonlinear_GB_Y_Z_W_diff_Ys), axis=0)
    our_method_mean = np.mean(np.array(our_method_diff_Ys), axis=0)
    assert linear_Y_Z_W_mean.shape == our_method_mean.shape

    # get error bars: std_err
    assert np.array(linear_Y_Z_W_diff_Ys).shape[0] == args.Y_coeffs_config_num

    def std_err(a):
        return np.std(a, axis=0) / np.sqrt(a.shape[0])

    linear_Y_Z_W_std_err = std_err(np.array(linear_Y_Z_W_diff_Ys))
    nonlinear_Y_Z_W_std_err = std_err(np.array(nonlinear_Y_Z_W_diff_Ys))
    nonlinear_RF_Y_Z_W_std_err = std_err(np.array(nonlinear_RF_Y_Z_W_diff_Ys))
    nonlinear_GB_Y_Z_W_std_err = std_err(np.array(nonlinear_GB_Y_Z_W_diff_Ys))
    our_method_std_err = std_err(np.array(our_method_diff_Ys))

    means_diff_Y = {'linear_Y_Z_W': linear_Y_Z_W_mean,
                    'nonlinear_Y_Z_W': nonlinear_Y_Z_W_mean,
                    'nonlinear_RF_Y_Z_W': nonlinear_RF_Y_Z_W_mean,
                    'nonlinear_GB_Y_Z_W': nonlinear_GB_Y_Z_W_mean,
                    'our_method': our_method_mean}
    std_err_diff_Y = {'linear_Y_Z_W': linear_Y_Z_W_std_err,
                      'nonlinear_Y_Z_W': nonlinear_Y_Z_W_std_err,
                      'nonlinear_RF_Y_Z_W': nonlinear_RF_Y_Z_W_std_err,
                      'nonlinear_GB_Y_Z_W': nonlinear_GB_Y_Z_W_std_err,
                      'our_method': our_method_std_err}

    means_diff_Y_df = pd.DataFrame(means_diff_Y)
    std_err_diff_Y_df = pd.DataFrame(std_err_diff_Y)

    Path(args.save_loc).mkdir(parents=True, exist_ok=True)
    means_diff_Y_df.to_csv(args.save_loc + \
                           "/unif_means_diff_Y_df{}.csv".format(args.Y_coeffs_config_num))
    std_err_diff_Y_df.to_csv(args.save_loc + \
                             "/unif_std_err_diff_Y_df{}.csv".format(args.Y_coeffs_config_num))

    print(std_err_diff_Y)
    plot_MSE_results(args, means_diff_Y, std_err_diff_Y)
