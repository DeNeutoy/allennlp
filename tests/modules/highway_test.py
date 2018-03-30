# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.modules import Highway, RecurrentHighway
from allennlp.common.testing import AllenNlpTestCase


class TestHighway(AllenNlpTestCase):
    def test_forward_works_on_simple_input(self):
        highway = Highway(2, 2)
        # pylint: disable=protected-access
        highway._layers[0].weight.data.fill_(1)
        highway._layers[0].bias.data.fill_(0)
        highway._layers[1].weight.data.fill_(2)
        highway._layers[1].bias.data.fill_(-2)
        input_tensor = Variable(torch.FloatTensor([[-2, 1], [3, -2]]))
        result = highway(input_tensor).data.numpy()
        assert result.shape == (2, 2)
        # This was checked by hand.
        assert_almost_equal(result, [[-0.0394, 0.0197], [1.7527, -0.5550]], decimal=4)


    def test_recurrent_highway_runs_forward(self):

        highway = RecurrentHighway(4)
        tensor1 = Variable(torch.randn([3,4]))
        tensor2 = Variable(torch.randn([3,4]))

        highway(tensor1, tensor2)

        tensor1 = Variable(torch.randn([3, 7, 4]))
        tensor2 = Variable(torch.randn([3, 7, 4]))

        highway(tensor1, tensor2)