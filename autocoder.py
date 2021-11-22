import torch.nn as nn

class Autocoder(nn.Module):
    def __init__(self):
        super(Autocoder,self).__init__()
        self.encoder = nn.Sequential(
            # 6300*8
            nn.Linear(9,5),
            nn.ReLU(),
            nn.Linear(5,4),
            nn.ReLU(),
            nn.Linear(4, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,4),
            nn.ReLU(),
            nn.Linear(4,5),
            nn.ReLU(),
            nn.Linear(5,9),
            # nn.Tanh()
        )

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode,decode

