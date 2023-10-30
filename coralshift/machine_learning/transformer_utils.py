import torch
import numpy as np
import torch.nn.functional as F

# from sktime.datasets import load_from_tsfile

from coralshift.machine_learning.data_handling.data_handler import DataHandler


def create_dummy_data(n_samples=100, n_features=3, n_timesteps=10, train=True):
    # Generate random data for the first n_samples/2 samples (greater than 100)
    data_high = np.random.randint(
        101, 200, size=(n_samples / 2, n_features, n_timesteps)
    )
    # Generate random data for the latter n_samples/2 samples (less than 10)
    data_low = np.random.randint(1, 10, size=(n_samples / 2, n_features, n_timesteps))
    # Concatenate the data arrays to get the final data array
    train_x = np.concatenate((data_high, data_low), axis=0)
    # Generate binary labels "yes" for the first n_samples/2 samples and "no" for the latter n_samples/2 samples
    train_y = np.array(["yes"] * n_samples / 2 + ["no"] * n_samples / 2)

    # Generate random data for the first n_samples/2 samples (greater than 100)
    data_high = np.random.randint(
        101, 200, size=(n_samples / 4, n_features, n_timesteps)
    )
    # Generate random data for the latter n_samples/2 samples (less than 10)
    data_low = np.random.randint(1, 10, size=(n_samples / 4, n_features, n_timesteps))
    # Concatenate the data arrays to get the final data array
    test_x = np.concatenate((data_high, data_low), axis=0)
    # Generate binary labels "yes" for the first 25 samples and "no" for the latter 25 samples
    test_y = np.array(["yes"] * n_samples / 4 + ["no"] * n_samples / 4)

    if train:
        return train_x, train_y
    else:
        return test_x, test_y


def get_data(train_path, test_path):
    torch.cuda.empty_cache()
    # outputs numpy arrays, shape: (n_samples, n_features, n_timesteps), (n_samples,)
    # train_x, train_y = load_from_tsfile(train_path, return_data_type="numpy3d")
    train_x, train_y = create_dummy_data(train=True)
    train_x = torch.tensor(train_x)
    # train_y contains thhe indices of the unique values found in train_y_orig
    train_y_orig, train_y = np.unique(train_y, return_inverse=True)
    # return number of classes
    n_values = np.max(train_y) + 1
    # on-hot encoding of labels
    train_y = np.eye(n_values)[train_y]

    # test_x, test_y = load_from_tsfile(test_path, return_data_type="numpy3d")
    test_x, test_y = create_dummy_data(train=False)
    test_x = torch.tensor(test_x)
    test_y_orig, test_y = np.unique(test_y, return_inverse=True)
    n_values = np.max(test_y) + 1
    test_y = np.eye(n_values)[test_y]
    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    dh = DataHandler()
    data_x = torch.concat((train_x, test_x), dim=0).permute(0, 2, 1)
    data_y = torch.concat((train_y, test_y), dim=0)
    dh.dataset_x = data_x
    dh.dataset_y = data_y
    return dh


def get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError(f"Activation should be relu/gelu, not {activation}.")
