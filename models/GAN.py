import torch
import torch.nn as nn
import numpy as np

try:
    import colored_traceback.always
except ImportError:
    pass


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout2d(0.4),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1)
        )
        
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(x.size()[0], -1)
        output = self.fc(conv)
        return output


class Reshape(nn.Module):
    def __init__(self, *out_size):
        super(Reshape, self).__init__()
        self.out_size = out_size
        
    def forward(self, x):
        x = x.view(x.size()[0], *self.out_size)
        return x
    
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(100, 3136),
            nn.BatchNorm1d(3136, momentum=0.9),
            nn.ReLU(),
            Reshape(64, 7, 7),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

    def forward(self, z):
        x = self.model(z)
        return x

    
def test_Discriminator():
    from torchsummary import summary
    input_size = (1, 28, 28)
    model = Discriminator()
    summary(model, input_size=input_size)
    
    # N = 10
    # x = torch.zeros((N, *input_size))
    # print(model(x))

    
def test_Generator():
    from torchsummary import summary
    input_size = (100,)
    model = Generator()
    summary(model, input_size=input_size)
    
    # N = 10
    # x = torch.zeros((N, *input_size))
    # print(model(x))

    
if __name__ == "__main__":
    test_Discriminator()
    test_Generator()
