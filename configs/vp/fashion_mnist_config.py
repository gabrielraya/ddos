
"""
Training NCSN++ on MNIST with VP SDE
Configuration regarding to the model with variance preserving SDE
"""

from configs.default_fashion_mnist_config import get_default_configs
import ml_collections


def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = "basic_sde"
    training.continuous = True
    training.reduce_mean = False

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'langevin'

    # data
    data = config.data
    data.centered = False  # works better without
    data.use_reduced_data = False
    data.sample_size = 0.1          # Training in a smaller subset

    # model
    model = config.model
    model.name = 'ncsnpp'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 256  # number of fourier features
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.embedding_type = 'fourier'
    model.init_scale = 0.
    model.fourier_scale = 30
    model.conv_size = 3

    return config

#%%
