import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from allennlp.common.tensor import sort_batch_by_length
from allennlp.modules import Seq2SeqEncoder


class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    """
    Pytorch's RNNs have two outputs: the hidden state for every time step, and the hidden state at
    the last time step for every layer.  We just want the first one as a single output.  This
    wrapper pulls out that output, and adds a :func:`get_output_dim` method, which is useful if you
    want to, e.g., define a linear + softmax layer on top of this to get some distribution over a
    set of labels.  The linear layer needs to know its input dimension before it is called, and you
    can get that from ``get_output_dim``.
    """
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        super(PytorchSeq2SeqWrapper, self).__init__()
        self._module = module

    def get_output_dim(self) -> int:
        return self._module.hidden_size * (2 if self._module.bidirectional else 1)

    def forward(self,
                inputs: torch.Tensor,
                sequence_lengths: torch.LongTensor = None) -> torch.Tensor:  # pylint: disable=arguments-differ

        if sequence_lengths is None:
            return self._module(inputs)[0]
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(inputs, sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs, sorted_sequence_lengths, batch_first=True)

        # Actually call the module on the sorted PackedSequence.
        packed_sequence_output, state = self._module(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
        # Restore the original indices and return the sequence.
        return unpacked_sequence_tensor[restoration_indices]
