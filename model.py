from torch.nn import Module, ReLU, Linear

from shared import DEVICE


class NeuralNet(Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        "Feed forward neural net with two hidden layers"
        super(NeuralNet, self).__init__()
        self.relu = ReLU()
        self.l1 = Linear(input_size, hidden_size, device=DEVICE)
        self.l2 = Linear(hidden_size, hidden_size, device=DEVICE)
        self.l3 = Linear(hidden_size, output_size, device=DEVICE)
        self.to(DEVICE)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No activation and No softmax
        return out
