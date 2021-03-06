import torch


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def create_model(config, sde):
    """
    Create the score model
    :param config: ymal file
    :return:
    """
    model_name = config.model.name
    score_model = get_model(model_name)(config, sde)
    return score_model


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde, model, train=False,continuous=True):
    """
    Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    :param sde: An `sde_lib.SDE` object that represents the forward SDE.
    :param model: A score model.
    :param train:  `True` for training and `False` for evaluation.
    :return: A score function
    """
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, t):
        t = t.clone().to(x.device).reshape((x.shape[0], ))
        score = model_fn(x, t)
        return score

    return score_fn