import torch

from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

class PytorchSeq2VecWrapper(Seq2VecEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the second one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.

    Also, there are lots of ways you could imagine going from an RNN hidden state at every
    timestep to a single vector - you could take the last vector at all layers in the stack, do
    some kind of pooling, take the last vector of the top layer in a stack, or many other  options.
    We just take the final hidden state vector, or in the case of a bidirectional RNN cell, we
    concatenate the forward and backward final states together. TODO(mattg): allow for other ways
    of wrapping RNNs.

    In order to be wrapped with this wrapper, a class must have the following members:

        - ``self.input_size: int``
        - ``self.hidden_size: int``
        - ``def forward(inputs: PackedSequence, hidden_state: torch.tensor) ->
          Tuple[PackedSequence, torch.Tensor]``.
        - ``self.bidirectional: bool`` (optional)

    This is what pytorch's RNN's look like - just make sure your class looks like those, and it
    should work.

    Note that we *require* you to pass sequence lengths when you call this module, to avoid subtle
    bugs around masking.  If you already have a ``PackedSequence`` you can pass ``None`` as the
    second parameter.
    """

    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        # Seq2VecEncoders cannot be stateful.
        super().__init__(stateful=False)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        try:
            is_bidirectional = self._module.bidirectional
        except AttributeError:
            is_bidirectional = False
        return self._module.hidden_size * (2 if is_bidirectional else 1)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor, hidden_state: torch.Tensor = None
    ) -> torch.Tensor:

        if mask is None:
            # If a mask isn't passed, there is no padding in the batch of instances, so we can just
            # return the last sequence output as the state.  This doesn't work in the case of
            # variable length sequences, as the last state for each element of the batch won't be
            # at the end of the max sequence length, so we have to use the state of the RNN below.
            return self._module(inputs, hidden_state)[0][:, -1, :]

        sequence_lengths = get_lengths_from_binary_sequence_mask(mask)
        zero_length_seqs = sequence_lengths == 0
        _, state = self.sort_and_run_forward(self._module, inputs, mask, hidden_state)

        # Deal with the fact the LSTM state is a tuple of (state, memory).
        if isinstance(state, tuple):
            state = state[0]

        # Pytorch's recurrent layers return state in the form
        # (num_layers * num_directions, batch_size, hidden_size) regardless
        # of the 'batch_first' flag, so we transpose and return them as a single
        # (batch_size, self.get_output_dim()) tensor.
        transposed_state = state.transpose(0, 1)
        broadcastable_zero_length_sequences = zero_length_seqs.view(-1, 1, 1)
        transposed_state.masked_fill_(broadcastable_zero_length_sequences, 0.0)

        # Extract the last hidden vector, including both forward and backward states
        # if the cell is bidirectional. Then reshape by concatenation (in the case
        # we have bidirectional states) or just squash the 1st dimension in the non-
        # bidirectional case. Return tensor has shape (batch_size, hidden_size * num_directions).
        try:
            last_state_index = 2 if self._module.bidirectional else 1
        except AttributeError:
            last_state_index = 1
        last_layer_state = transposed_state[:, -last_state_index:, :]
        return last_layer_state.contiguous().view([-1, self.get_output_dim()])
