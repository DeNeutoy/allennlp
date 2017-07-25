import torch
from torch.nn.utils.rnn import pack_padded_sequence

from allennlp.common.tensor import sort_batch_by_length
from allennlp.modules import Seq2VecEncoder


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
    We just take the final hidden state vector.  TODO(mattg): allow for other ways of wrapping
    RNNs.
    """
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        super(PytorchSeq2VecWrapper, self).__init__()
        self._module = module

    def get_output_dim(self) -> int:
        return self._module.hidden_size * (2 if self._module.bidirectional else 1)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                sequence_lengths: torch.LongTensor = None) -> torch.Tensor:

        if sequence_lengths is None:
            return self._module(inputs)[0][:, :, -1]
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(inputs,
                                                                                           sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs, sorted_sequence_lengths, batch_first=True)

        # Actually call the module on the sorted PackedSequence.
        _, state = self._module(packed_sequence_input)

        # Deal with the fact the LSTM state is a tuple of (state, memory)
        if isinstance(state, tuple):
            return state[0][restoration_indices]
        # Restore the original indices and return the final state.
        return state[restoration_indices]
