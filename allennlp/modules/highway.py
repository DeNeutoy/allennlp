"""
A `Highway layer <https://arxiv.org/abs/1505.00387>`_ that does a gated combination of a linear
transformation and a non-linear transformation of its input.
"""

from typing import Callable

import torch
from overrides import overrides


class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, to we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.nn.functional.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class RecurrentHighway(torch.nn.Module):
    """
    This module will apply a recurrent highway layer to its two inputs,
    typically the input and output to some other function which you want to gate.
            gate = sigmoid(W_i * x + W_o * y)
            output = gate * y + (1 - gate) * (W_i2 * x)

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x` and :math:`y`. We assume the input has shape ``(batch_size,
        input_dim)``.
    """
    def __init__(self, input_dim: int) -> None:
        super(RecurrentHighway, self).__init__()
        self._input_dim = input_dim
        self._gate_projection = torch.nn.Linear(input_dim * 2, input_dim * 2)
        self._linear_projection = torch.nn.Linear(input_dim, input_dim)
        # We should bias the highway layer to just carry its input forward.  We do that by
        # setting the bias on the gate projection to be negative, because then when we add
        # the two projections (one from the projected input, one from the projected output)
        # we will get a bias of -1, making the sigmoid biased towards small values, meaning
        # we bias towards passing through a linear projection of the input.
        self._gate_projection.bias.data.fill_(-0.5)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                pre_layer_tensor: torch.Tensor,
                post_layer_tensor: torch.Tensor) -> torch.Tensor:

        gate_projection = self._gate_projection(torch.cat([pre_layer_tensor, post_layer_tensor], -1))
        # NOTE: if you modify this, think about whether you should modify the initialization
        # above, too.
        gate = sum(gate_projection.split(self._input_dim, -1))
        gate = torch.nn.functional.sigmoid(gate)

        return gate * post_layer_tensor + (1 - gate) * self._linear_projection(pre_layer_tensor)
