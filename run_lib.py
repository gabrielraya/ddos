
import os
import tensorflow as tf
from absl import flags
from torch.utils import tensorboard


FLAGS = flags.FLAGS

def train(config, workdir):
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model
    # score_model = mul

#%%

