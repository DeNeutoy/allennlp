import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.stacked_alternating_lstm_cuda import HighwayLSTM


class TestCustomHighwayLSTM(AllenNlpTestCase):

    def test_small_model(self):
        args = self.get_models_and_inputs(5, 3, 11, 2, 5, 0.0)
        self.forward_and_backward_outputs_match(*args)

    def test_large_model(self):
        args = self.get_models_and_inputs(83, 103, 311, 8, 101, 0.0)
        self.forward_and_backward_outputs_match(*args)

    def test_validation_forward_pass_is_deterministic_in_model_with_dropout(self):

        _, model, _, model_input, lengths = self.get_models_and_inputs(5, 3, 11, 2, 5, dropout_prob=0.5)
        model.eval()
        model_input = pack_padded_sequence(model_input, lengths, batch_first=True)
        output, _ = model(model_input)
        output, _ = pad_packed_sequence(output, batch_first=True)

        for i in range(3):
            output_new, _ = model(model_input)
            output_new, _ = pad_packed_sequence(output_new, batch_first=True)
            diff = torch.max(output.data - output_new.data)
            assert diff < 1e-4, "forward pass is not deterministic in validation mode."
            output = output_new

    @staticmethod
    def forward_and_backward_outputs_match(baseline_model, kernel_model,
                                           baseline_input, kernel_input, lengths):

        packed_baseline_input = pack_padded_sequence(baseline_input, lengths, batch_first=True)
        baseline_output, _ = baseline_model(packed_baseline_input)
        baseline_output, _ = pad_packed_sequence(baseline_output, batch_first=True)

        packed_kernel_input = pack_padded_sequence(kernel_input, lengths, batch_first=True)
        kernel_output, _ = kernel_model(packed_kernel_input)
        kernel_output, _ = pad_packed_sequence(kernel_output, batch_first=True)

        diff = torch.max(baseline_output.data - kernel_output.data)
        assert diff < 1e-4, "Output does not match: " + str(diff)

        # Backprop some random error.
        random_error = torch.randn(baseline_output.size()).cuda()
        baseline_model.zero_grad()
        baseline_output.backward(random_error)

        kernel_model.zero_grad()
        kernel_output.backward(random_error)
        
        input_grad_diff = torch.max(baseline_input.grad.data - kernel_input.grad.data)
        assert input_grad_diff < 1e-4, "Input grad does not match: " + str(input_grad_diff)

        weight_index = 0
        bias_index = 0
        for layer in range(baseline_model.num_layers):
            input_grad = getattr(baseline_model, 'layer_%d' % layer).input_linearity.weight.grad
            state_grad = getattr(baseline_model, 'layer_%d' % layer).state_linearity.weight.grad
            bias_grad = getattr(baseline_model, 'layer_%d' % layer).state_linearity.bias.grad

            kernel_input_grad = kernel_model.weight.grad[weight_index: weight_index+input_grad.nelement()].view(input_grad.size(1), input_grad.size(0)).t()
            weight_index += input_grad.nelement()

            mine_h_grad = kernel_model.weight.grad[weight_index: weight_index + state_grad.nelement()].view(state_grad.size(1), state_grad.size(0)).t()
            weight_index += state_grad.nelement()

            mine_bias = kernel_model.bias.grad[bias_index:bias_index+bias_grad.nelement()]
            bias_index += bias_grad.nelement()

            x_diff = torch.max(kernel_input_grad.data - input_grad.data)
            assert x_diff < 1e-4, "Layer %d x_weight does not match: " % layer + str(x_diff)

            h_diff = torch.max(mine_h_grad.data - state_grad.data)
            assert h_diff < 1e-4, "Layer %d h_weight does not match: " % layer + str(h_diff)

            bias_diff = torch.max(mine_bias.data - bias_grad.data)
            assert bias_diff < 1e-4, "Layer %d bias does not match: " % layer + str(bias_diff)

    @staticmethod
    def get_models_and_inputs(batch_size, input_size, output_size, num_layers, timesteps, dropout_prob):

        baseline = StackedAlternatingLstm(input_size, output_size, num_layers, dropout_prob).cuda()
        kernel_version = HighwayLSTM(input_size, output_size, num_layers, dropout_prob).cuda()

        # Copy weights from non-cuda version into cuda version,
        # so we are starting from exactly the same place.
        weight_index = 0
        bias_index = 0
        for layer_index in range(num_layers):

            layer = getattr(baseline, 'layer_%d' % layer_index)
            input_weight = layer.input_linearity.weight
            state_weight = layer.state_linearity.weight
            bias = layer.state_linearity.bias

            kernel_version.weight.data[weight_index: weight_index + input_weight.nelement()].copy_(input_weight.data.t())
            weight_index += input_weight.nelement()

            kernel_version.weight.data[weight_index: weight_index + state_weight.nelement()].copy_(state_weight.data.t())
            weight_index += state_weight.nelement()

            kernel_version.bias.data[bias_index:bias_index + bias.nelement()].copy_(bias.data)
            bias_index += bias.nelement()

        input = torch.randn(batch_size, timesteps, input_size).cuda()
        # Clone variable so different models are
        # completely separate in the graph.
        input2 = input.clone()
        baseline_input = Variable(input, requires_grad=True)
        kernel_version_input = Variable(input2, requires_grad=True)
        lengths = [timesteps - int((i / 2)) for i in range(batch_size)]
        lengths = lengths[:batch_size]

        return baseline, kernel_version, baseline_input, kernel_version_input, lengths
