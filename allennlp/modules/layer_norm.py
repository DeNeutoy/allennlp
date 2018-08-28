import torch


class LayerNorm(torch.nn.Module):
    # pylint: disable=line-too-long
    """
    An implementation of `Layer Normalization
    <https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5>`_ .

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    Parameters
    ----------
    dimension : ``int``, required.
        The dimension of the layer output to normalize.
    eps : ``float``, optional, (default = 1e-6)
        An epsilon to prevent dividing by zero in the case
        the layer has zero variance.
    moving_average : ``float``, optional (default = None)
        The rate at which to include current values into the
        moving average
    Returns
    -------
    The normalized layer output.
    """
    def __init__(self,
                 dimension: int,
                 eps: float = 1e-6,
                 moving_average: float = None) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))
        self.eps = eps
        self.moving_average = moving_average

        if self.moving_average is not None:
            self.register_buffer("gamma_moving_avg", self.gamma.data)
            self.register_buffer("beta_moving_avg", self.beta.data)

    def forward(self, tensor: torch.Tensor):  # pylint: disable=arguments-differ
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, unbiased=False, keepdim=True)

        if self.moving_average is not None:
            momentum = self.moving_average
            gamma = (1.0 - momentum) * self.gamma + momentum * self.gamma_moving_avg
            self.gamma_moving_avg = gamma.data
            beta = (1.0 - momentum) * self.beta_moving_avg + momentum * self.beta
            self.beta_moving_avg = beta.data
        else:
            gamma = self.gamma
            beta = self.beta
        return gamma * (tensor - mean) / (std + self.eps) + beta
