import torch.nn as nn

class Autocoder(nn.Module):
    def __init__(self):
        super(Autocoder,self).__init__()
        self.encoder = nn.Sequential(
            # 6300*8
            nn.Linear(8, 6),
            nn.Sigmoid(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 6),
            nn.Sigmoid(),
            nn.Linear(6, 8),
            nn.Tanh()
        )

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode,decode

