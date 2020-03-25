from torch import nn



class simple_ae(nn.Module):
    def __init__(self, inp_size):
        super(simple_ae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inp_size, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True), nn.Linear(8, 4), nn.ReLU(True), nn.Linear(4, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(True),
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True), nn.Linear(16, inp_size), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x