import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


class ImgSimPert_data_by_cov(Dataset):
    """ Image Perturbation dataset class
    for phi model training

     Input:
     - dataset_path (path to load data from)
     - Z_dim (dimensions of covariates Z)
     - W_dim (dimensions of covariates W)
     - X_dim (dimensions of covariate X)
     - Y_dim (dimensions of covariate Y) """

    def __init__(self, dataset_path,
                 Z_dim, W_dim,
                 X_dim, Y_dim):
        read_in = np.load(dataset_path)
        idx = read_in.files[0]
        self.img_sim = read_in[idx]
        self.Z = torch.tensor(self.img_sim
                              [:, :Z_dim]).float()
        self.W = torch.tensor(self.img_sim
                              [:, Z_dim:
                                  Z_dim + W_dim]).float()
        self.X = torch.tensor(self.img_sim
                              [:, Z_dim + W_dim:
                                  Z_dim + W_dim + X_dim]).float()
        self.Y = torch.tensor(self.img_sim
                              [:, Z_dim + W_dim + X_dim:
                                  Z_dim + W_dim + X_dim + Y_dim]).float()
        self.phis = torch.tensor(self.img_sim
                              [:, Z_dim + W_dim + X_dim + Y_dim:]).float()

        print("self.img_sim.shape: ", self.img_sim.shape)
        print("self.Z.shape: ", self.Z.shape)
        print("self.W.shape: ", self.W.shape)
        print("self.X.shape: ", self.X.shape)
        print("self.Y.shape: ", self.Y.shape)
        print("self.phis.shape: ", self.phis.shape)

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = index.cpu().numpy()
        Z = self.Z[index]
        W = self.W[index]
        X = self.X[index]
        Y = self.Y[index]
        phis = self.phis[index]
        return Z, W, X, Y, phis, index

    def __len__(self):
        return len(self.img_sim)


def get_img_sim_loaders_by_cov(args,
                               split='all',
                               **kwargs):
    """ helper function to load Image Perturbation
    dataset for phis model training.

     Input:
     - args (run configs from user)
     - split (flag to specify split of interest)
    Output:
    - loader (loader for relevant splot by batch) """
    if split == 'train_seen_train':
        train_loader = DataLoader(ImgSimPert_data_by_cov(args.data_save_dir + "_trainset_seen_train.npz",
                                                         args.Z_dim, args.W_dim,
                                                         args.X_dim, args.Y_dim),
                                  batch_size=args.train_batch_size, **kwargs)
        return train_loader

    if split == 'train_seen_test':
        train_loader = DataLoader(ImgSimPert_data_by_cov(args.data_save_dir + "_trainset_seen_test.npz",
                                                         args.Z_dim, args.W_dim,
                                                         args.X_dim, args.Y_dim),
                                  batch_size=args.train_batch_size, **kwargs)
        return train_loader

    elif split == 'test':
        test_loader = DataLoader(ImgSimPert_data_by_cov(args.data_save_dir + "_testset.npz",
                                                        args.Z_dim, args.W_dim,
                                                        args.X_dim, args.Y_dim),
                                 batch_size=args.test_batch_size, **kwargs)
        return test_loader
    elif split == 'test_unseen':
        test_loader = DataLoader(ImgSimPert_data_by_cov(args.data_save_dir + "_testset_unseen.npz",
                                                        args.Z_dim, args.W_dim,
                                                        args.X_dim, args.Y_dim),
                                 batch_size=args.test_batch_size, **kwargs)
        return test_loader
    elif split == 'test_unseen_train':
        test_loader = DataLoader(ImgSimPert_data_by_cov(args.data_save_dir + "_testset_unseen_train.npz",
                                                        args.Z_dim, args.W_dim,
                                                        args.X_dim, args.Y_dim),
                                 batch_size=args.test_batch_size, **kwargs)
        return test_loader
    elif split == 'test_unseen_test':
        test_loader = DataLoader(ImgSimPert_data_by_cov(args.data_save_dir + "_testset_unseen_test.npz",
                                                        args.Z_dim, args.W_dim,
                                                        args.X_dim, args.Y_dim),
                                 batch_size=args.test_batch_size, **kwargs)
        return test_loader
    elif split == 'all':
        all_loader = DataLoader(ImgSimPert_data_by_cov(args.data_save_dir + "_all.npz",
                                                       args.Z_dim, args.W_dim,
                                                       args.X_dim, args.Y_dim),
                                batch_size=args.train_batch_size, **kwargs)
        return all_loader
    else:
        raise NotImplementedError


def get_full_vector_img_sim(args, split='all'):
    """ helper function to load values for target of interest
    (i.e. covariate of interest) for entire split of interest.

     Input:
     - args (run configs from user)
     - split (split of interest to obtain values from)
    Output:
    - Z, W, X, Y, phis (covariates from dataset) """

    if split == 'all':
        dataset_path = args.data_save_dir + "_all.npz"
    elif split == 'train_seen_train':
        dataset_path = args.data_save_dir + "_trainset_seen_train.npz"
    elif split == 'train_seen_test':
        dataset_path = args.data_save_dir + "_train_seen_test.npz"
    elif split == 'test':
        dataset_path = args.data_save_dir + "_testset.npz"
    elif split == 'test_unseen':
        dataset_path = args.data_save_dir + "_testset_unseen.npz"
    elif split == 'test_unseen_train':
        dataset_path = args.data_save_dir + "_testset_unseen_train.npz"
    elif split == 'test_unseen_test':
        dataset_path = args.data_save_dir + "_testset_unseen_test.npz"
    else:
        raise NotImplementedError

    read_in = np.load(dataset_path)
    idx = read_in.files[0]
    img_sim_df = read_in[idx]

    Z = torch.tensor(img_sim_df
                     [:, :args.Z_dim]).float()
    W = torch.tensor(img_sim_df
                     [:, args.Z_dim:
                         args.Z_dim + args.W_dim]).float()
    X = torch.tensor(img_sim_df
                     [:, args.Z_dim + args.W_dim:
                         args.Z_dim + args.W_dim + args.X_dim]).float()
    Y = torch.tensor(img_sim_df
                     [:, args.Z_dim + args.W_dim + args.X_dim:
                         args.Z_dim + args.W_dim + args.X_dim + args.Y_dim]).float()
    phis = torch.tensor(img_sim_df
                     [:, args.Z_dim + args.W_dim + args.X_dim + args.Y_dim:]).float()

    return Z, W, X, Y, phis
