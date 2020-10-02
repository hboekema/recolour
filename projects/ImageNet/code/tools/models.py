
""" File for retrieving architecture inputs, outputs, and losses. """


from tensorflow.keras.losses import MeanSquaredError

from architectures.generator.UNetFCNN import UNetFCNN
from architectures.generator.SmallUNetFCNN import SmallUNetFCNN
from architectures.generator.FCNN import FCNN
from architectures.generator.WGAN_FCNN import WGAN_FCNN
from architectures.generator.MLP import MLP

from architectures.discriminator.SimpleDCNN import SimpleDCNN


def get_architecture_inputs_outputs(ARCHITECTURE, PARAMS):
    """Setup the chosen architecture with the given parameters, returning the inputs and outputs needed to build a Keras Model instance.

    Parameters
    ----------
    ARCHITECTURE : str
        Architecture name
    PARAMS : dict
        Parameters required by chosen architecture

    Returns
    -------
    list
        Inputs and outputs to the Model

    Raises
    ------
    ValueError
        If architecture not implemented
    """

    if ARCHITECTURE == "UNetFCNN":
        return UNetFCNN(**PARAMS)
    elif ARCHITECTURE == "SmallUNetFCNN":
        return SmallUNetFCNN(**PARAMS)
    elif ARCHITECTURE == "FCNN":
        return FCNN(**PARAMS)
    elif ARCHITECTURE == "WGAN_FCNN":
        return WGAN_FCNN(**PARAMS)
    elif ARCHITECTURE == "MLP":
        return MLP(**PARAMS)
    elif ARCHITECTURE == "SimpleDCNN":
        return SimpleDCNN(**PARAMS)
    else:
        raise ValueError("Architecture '{}' not implemented.".format(ARCHITECTURE))


def get_architecture_loss(ARCHITECTURE, LOSS_WEIGHTS=[1.0]):
    """Get the loss functions and full loss weights for the chosen architecture.

    Parameters
    ----------
    ARCHITECTURE : str
        Architecture name
    LOSS_WEIGHTS : list
        Relative loss weights for the losses for this architecture

    Returns
    -------
    list
        Loss functions and loss weights for the architecture outputs
    """

    # TODO: Currently, the ARCHITECTURE parameter is not used. Re-implement when this parameter is needed.

    loss = [MeanSquaredError()]

    return loss, LOSS_WEIGHTS

