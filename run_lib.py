
import os
import tensorflow as tf
from absl import flags
from torch.utils import tensorboard
from models import utils as mutils

import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

from models import ncsnpp

FLAGS = flags.FLAGS

def train(config, workdir):
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model
    score_model = mutils.create_model(config)
    print(score_model)
    random_t = torch.rand(128)
    print(random_t)
    print(score_model(random_t))
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    # model = torchvision.models.resnet50(False)
    # # Have ResNet model take in grayscale rather than RGB
    # model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # images, labels = next(iter(trainloader))
    # grid = torchvision.utils.make_grid(images)
    # writer.add_image('images', grid, 0)
    # writer.add_graph(model, images)
    # #
    # for n_iter in range(100):
    #     writer.add_scalar('Loss/train', np.random.random(), n_iter)
    #     writer.add_scalar('Loss/test', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    #     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    writer.close()
#%%








