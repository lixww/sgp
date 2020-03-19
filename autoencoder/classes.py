from torch import nn



class simple_ae(nn.Module):
    def __init__(self, img_size):
        super(simple_ae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(img_size, 10),
            nn.ReLU(True),
            nn.Linear(10, 5),
            nn.ReLU(True), nn.Linear(5, 15), nn.ReLU(True), nn.Linear(15, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 15),
            nn.ReLU(True),
            nn.Linear(15, 5),
            nn.ReLU(True),
            nn.Linear(5, 10),
            nn.ReLU(True), nn.Linear(10, img_size), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x