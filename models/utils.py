

def create_model(config):
    """
    Create the score model
    :param config: ymal file
    :return:
    """
    model_name = config.model.name
    score_model = get_model()