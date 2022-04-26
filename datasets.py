# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""

import torchvision.transforms as transforms
import torchvision.datasets as tds
import torch


def get_dataset(config, evaluation=False):
    """
    Create data loaders for training and evaluation
    :param config: a ml_collection.ConfigDict parsed from config files
    :param evaluation: if `True`, fix number of epochs to 1
    :return: train_ds, eval_ds, datset_builder
    """

    # Compute batch size for this worker
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size

    num_epochs = None if not evaluation else 1

    if config.data.dataset == 'MNIST':
        train_ds = tds.MNIST('data/', train=True, transform=transforms.ToTensor(), download=True)
        eval_ds = tds.MNIST('data/', train=False, transform=transforms.ToTensor(), download=True)

    elif config.data.dataset == 'FashionMNIST':
        train_ds = tds.FashionMNIST('data/', train=True, transform=transforms.ToTensor(), download=True)
        eval_ds = tds.FashionMNIST('data/', train=False, transform=transforms.ToTensor(), download=True)


        if config.data.use_reduced_data:
            print('Original dataset size: ', len(train_ds))
            indices = list(range(0, 2 * int(len(train_ds) * config.data.sample_size), 2))
            train_ds = torch.utils.data.Subset(train_ds, indices)
            print('Reduced dataset size: ', len(train_ds))

            sample_size=0.2
            print('Original test dataset size: ', len(eval_ds))
            indices = list(range(0, 2 * int(len(eval_ds) * sample_size), 2))
            eval_ds = torch.utils.data.Subset(eval_ds, indices)
            print('Reduced dataset size: ', len(eval_ds))

    return train_ds, eval_ds



def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x