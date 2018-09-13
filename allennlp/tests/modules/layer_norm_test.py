
import torch

from allennlp.modules.layer_norm import LayerNorm


class TestLayerNorm:

    def test_layer_norm_can_use_running_avg(self):

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = LayerNorm(10, moving_average=0.99)
                self.linear = torch.nn.Linear(10, 10)
            def forward(self, x):
                return self.norm(self.linear(x)) + x

        net = Net()
        opt = torch.optim.SGD(net.parameters(), 0.1)
        for i in range(200):
            data = torch.randn([5, 10])
            x = net(data)
            opt.zero_grad()
            loss = (x - data).pow(2).sum()
            loss.backward()
            opt.step()
            print("beta ", net.norm.beta_moving_avg)
            print("gamma ", net.norm.gamma_moving_avg)
