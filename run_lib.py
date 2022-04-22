
import os
import tensorflow as tf
from absl import flags
from torch.utils import tensorboard

import losses
import sde_lib
from models import utils as mutils
import datasets

import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import logging
from models import ncsnpp

FLAGS = flags.FLAGS

def train(config, workdir):
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(config)

    # Setup SDEs
    if config.training.sde.lower() == "basic_sde":
        sde = sde_lib.BASIC_SDE(sigma=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Initialize model
    score_model = mutils.create_model(config, sde)
    score_model = torch.nn.DataParallel(score_model)
    score_model = score_model.to(config.device)

    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    initial_step = int(state['step'])

    # Build one-step training and evaluation functions
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimizer)


    data_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=4)

    tqdm_epoch = tqdm.trange(config.training.n_iter)

    loss_fn = losses.get_sde_loss_fn(sde)

    print("\nStart training of score model\n")

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in tqdm_epoch: #num_train_steps+1):

        for x, y in data_loader:
            x = x.to(config.device)
            # Execute one training step
            # loss = train_step_fn(state, x)
            loss = loss_fn(score_model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the averaged training loss so far.
        logging.info("epoch: %d, training_loss: %.5e" % (epoch, loss.item()))
        writer.add_scalar("training_loss", loss, epoch)
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(loss.item()))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), workdir+'/ckpt.pth')

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    # print(loss)

    # x, labels = next(iter(trainloader))
    # print("Image shape:", x.shape)
    # random_t = torch.rand(x.shape[0])
    # print("H1+t", score_model(x, random_t).shape)
    # summary(score_model, (1,28,28), random_t)

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








