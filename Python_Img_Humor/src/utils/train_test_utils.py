from pathlib import Path
import matplotlib.pyplot as plt
import torch
import logging
from sklearn.metrics import explained_variance_score
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from data import get_full_vector_img_sim, get_img_sim_loaders_by_cov, \
    get_humicroedit_loaders_by_cov, get_full_vector_humicroedit


def train_phi_models(args, model, loss_criterion, optimizer,
                     scheduler, train_loader, nested_model=False):
    """ main objective function training routine (full objective).

     Input:
     - args (config from user),
     - model (model to be trained),
     - loss_criterion (predefined loss criterion fit to problem, set
     in run_phi_model.py:train_phis())
     - optimizer (predefined optimizer object for training, set
     in run_phi_model.py:train_phis()),
     - scheduler (potential scheduler object to deploy in training)
     - train_loader (loader of training dataset by batches)
     - nested_model (flag indicating if phis~Z, i.e, nested, or phis~Z,W)
     Output:
     - losses (array containing all batch losses),
     - ex_vars (explained variance results) """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    losses = []
    ex_vars = []
    for batch_idx, (Z, W, X, Y, phis, idx) in enumerate(train_loader):
        Z, W, X, Y, phis = \
            Z.to(device), W.to(device), X.to(device), \
            Y.to(device), phis.to(device)
        optimizer.zero_grad()

        if args.dataset == 'PertImgSim':
            num_phis = args.Z_dim // (args.window_size_phi ** 2)
        elif (args.dataset == 'Humicroedit'):
            num_phis = args.num_phis
        else:
            raise NotImplementedError

        ex_var = []
        if not nested_model:
            combined_input = torch.cat((Z, W), axis=1)
            phi_hats = model(combined_input)
        else:
            phi_hats = model(Z)
        for i in range(num_phis):
            real_phi_i = phis[:, i].unsqueeze(dim=1)
            pred_phi_i = phi_hats[i]
            if i == 0:
                loss = loss_criterion(pred_phi_i, real_phi_i)
            else:
                loss += loss_criterion(pred_phi_i, real_phi_i)
            ex_var.append(explained_variance_score(pred_phi_i.detach().cpu().numpy(),
                                                   real_phi_i.detach().cpu().numpy()))

        loss.backward(retain_graph=True)
        losses.append(loss.detach().cpu().numpy())
        ex_vars.append(np.mean(ex_var))
        optimizer.step()
        scheduler.step()
    return losses, ex_vars


def obtain_phi_predictions(args,
                           phi_models, train=False):
    """ training of Y prediction model

     Input:
     - args (config from user),
     - phi_models (trained phis model),
     - train (flag to indicate if called from train or test)
     Output:
     - Gs (phi predictions) """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")

    if args.dataset == 'PertImgSim':
        loading_fn = get_img_sim_loaders_by_cov
    elif args.dataset == 'Humicroedit':
        loading_fn = get_humicroedit_loaders_by_cov
    else:
        raise NotImplementedError

    if train:
        train_loader = loading_fn(args=args, split='train_seen_train')
    else:
        train_loader = loading_fn(args=args, split='test_unseen_test')

    Gs = []
    ex_vars = []
    phis_l = []
    for batch_idx, (Z, W, X, Y, phis, idx) in enumerate(train_loader):
        Z, W = \
            Z.to(device), W.to(device)

        combined_input = torch.cat((Z, W), axis=1)
        G = phi_models(combined_input)
        if batch_idx == 0:
            first_g = [G[i][1].item() for i in range(len(G))]

        G = torch.cat(G).detach().cpu().numpy().reshape(-1, Z.shape[0])
        G = np.transpose(G)
        Gs.append(G)
        phis_l.append(phis)
        ex_vars.append(explained_variance_score(phis, G))
    Gs = np.concatenate(Gs)
    phis_l = np.concatenate(phis_l)
    test_G = first_g == Gs[1, :]
    assert torch.tensor(test_G).all()
    if train:
        print("train G")
        print("=======")
        print("mean ex var: ", np.mean(ex_vars))
        print("total ex var: ", explained_variance_score(phis_l, Gs))
    else:
        print("test G")
        print("=======")
        print("mean ex var: ", np.mean(ex_vars))
        print("total ex var: ", explained_variance_score(phis_l, Gs))
    return Gs


def train_y_model(args, phi_models):
    """ training of Y prediction model

     Input:
     - args (config from user),
     - phi_models (trained phis model),
     Output:
     - model_y (trained model y) """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'PertImgSim':
        Z, W, X, Y, phis = get_full_vector_img_sim(args, split='train_seen_train')
        num_phis = args.Z_dim // (args.window_size_phi ** 2)
    elif args.dataset == 'Humicroedit':
        Z, W, X, Y, phis = get_full_vector_humicroedit(args, split='train_seen_train')
        num_phis = args.num_phis
    else:
        raise NotImplementedError

    Gs = obtain_phi_predictions(args, phi_models, train=True)
    print("Gs.shape: ", Gs.shape)
    assert Gs.shape == (args.dataset_length, num_phis)

    if args.Y_target == "continuous":
        lasso = linear_model.Lasso()
        predictor = GridSearchCV(lasso,
                                 {'alpha': list(map(float, args.alphas))})
    else:
        raise NotImplementedError
    predictor.fit(Gs,
                  Y.detach().cpu().numpy().ravel())
    print("predictor best lambda: ", predictor.best_params_)
    model_y = predictor.best_estimator_
    return model_y


def test_y_model(args, phi_models, model_y):
    """ testing of Y prediction model

     Input:
     - args (config from user),
     - phi_models (trained phis model),
     - model_y (trained model y)
     Output:
     - Y_hat (vector of predictions Y_hat),
     - (model_y.coef_, model_y.intercept_) (tuple of weights of fitted model)
     """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'PertImgSim':
        num_phis = args.Z_dim // (args.window_size_phi ** 2)
    elif (args.dataset == 'Humicroedit') or (args.dataset == 'Genomic'):
        num_phis = args.num_phis

    Gs = obtain_phi_predictions(args, phi_models, train=False)
    assert Gs.shape == (args.dataset_length_test, num_phis)

    if args.Y_target == 'continuous':
        Y_hat = model_y.predict(Gs)
        return Y_hat, (model_y.coef_, model_y.intercept_)
    else:
        raise NotImplementedError


def train(model, device, optimizer,
          train_loader, loss_criterion,
          scheduler=None):
    """ Train given model in batches.

     Input:
     - model (model to train),
     - device (device to be used),
     - optimizer (optimizer initialized with model's parameters),
     - train_loader (loader to iterate over batches from),
     - loss_criterion (form of objective function)
     - scheduler (optional, if passed, scheduler object)
     - option: toggle lr scheduler as needed """
    for batch_idx, (data, target, idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        #     scheduler.step()


def test(model, device, test_loader,
         loss_criterion, classification):
    """ Test given model in batches.

     Input:
     - model (model to test),
     - device (device to be used),
     - test_loader (loader to iterate over batches from),
     - loss_criterion (form of objective function)
     - classification (flag to indicate whether the model is class. or reg.
     this will matter for reporting accuracy/variance exp.)
     Output: test_loss (avg. loss across batches, optional) """
    test_loss = 0
    k = 0
    if classification:
        correct = 0
    else:
        outputs = []
        targets = []
    with torch.no_grad():
        for batch_idx, (data, target, *_) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += loss_criterion(output, target).item()
            # classification perf.: count correct pred.
            if classification:
                pred = output > 0.5
                if batch_idx == 0:
                    k = target.shape[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            # regression perf.: accum. outputs for var. exp.
            else:
                if output.shape[1] > 1:
                    if type(outputs) == list:
                        outputs = output.cpu().numpy()
                        targets = target.cpu().numpy()
                    else:
                        outputs = np.concatenate((outputs,
                                                  output.cpu().numpy()))
                        targets = np.concatenate((targets,
                                                  target.cpu().numpy()))
                else:
                    outputs.append(output.cpu().numpy()[0])
                    targets.append(target.cpu().numpy()[0])
    # avg. batch losses
    test_loss /= len(test_loader.dataset)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Test results ...')

    if classification:
        logger.info('\nTest on 20% train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, k * len(test_loader.dataset),
                                100. * (correct / (k * len(test_loader.dataset)))))

    else:
        logger.info('\nTest set: Average loss: {:.4f}, explained var.: {:.4f}\n'.format(
            test_loss, explained_variance_score(targets, outputs)))


def visualize(args, model, device,
              test_loader, model_name):
    """ Helper function: visualizing predictions vs. ground truth
    via scatter plot for a given trained model.

     Input:
     - args (run config),
     - model (trained model to visualize prediction of)
     - device (device to be used, i.e. GPU or not)
     - test_loader (object to load test batches to get predictions for)
     - model_name (string representing the type of model in question) """
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target, *_) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            outputs.append(output.cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy())
    outputs = np.array(outputs, dtype=object)
    targets = np.array(targets, dtype=object)
    if model_name == 'y_model':
        visualize_ATEs(Xs=outputs, Ys=targets,
                       x_name='y preds.',
                       y_name="y targets",
                       save_loc=args.save_loc,
                       save_name="y_scatter_NN")
    elif model_name == 'w_model':
        visualize_ATEs(Xs=outputs, Ys=targets,
                       x_name='w preds.',
                       y_name="w targets",
                       save_loc=args.save_loc,
                       save_name="w_scatter_NN")


def visualize_ATEs(Xs, Ys,
                   x_name, y_name,
                   save_loc, save_name):
    """ helper function to create and save scatter plots,
    for some arrays of interest, Xs and Ys.

     Input:
     - Xs (values to plot on X axis)
     - Ys (values to plot on Y axis)
     - x_name (label for X axis)
     - y_name (label for Y axis)
     - save_loc (path to save plot)
     - save_name (name to save plot) """
    plt.figure()
    Xs = Xs.flatten()
    Ys = Ys.flatten()
    df = pd.DataFrame({x_name: Xs,
                       y_name: Ys})
    ax = sns.scatterplot(x=x_name, y=y_name, data=df)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    start_ax_range = min(xmin, ymin) - 0.1
    end_ax_range = max(xmax, ymax) + 0.1
    ax.set_xlim(start_ax_range, end_ax_range)
    ax.set_ylim(start_ax_range, end_ax_range)
    ident = [start_ax_range, end_ax_range]
    plt.plot(ident, ident, '--')

    Path(save_loc).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_loc + '/' + save_name + '.png',
                bbox_inches='tight')
