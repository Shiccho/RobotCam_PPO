import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.distributions import Categorical
from environment import Env

class ActorCriticNet(nn.Module):
    def __init__(self, batch_size = 32, t_num=4):
        super().__init__()
        self.batch_size = batch_size
        self.resnet = ResNet(pretrained=True)
        self.flatten = nn.Flatten()
        self.bilstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc_pi = nn.Linear(512, 12)
        self.fc_v = nn.Linear(512, 1)
        self.t_num = t_num
    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = x.reshape(-1, self.t_num, 512)
        bilstm_out, bilstm_hc = self.bilstm(x)
        x = torch.cat([bilstm_hc[0][0], bilstm_hc[0][1]], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        pi_logits = self.fc_pi(x)
        value = self.fc_v(x)
        #print('value: ', value[0].item())
        return pi_logits, value

class Discriminator(nn.Module):
    def __init__(self, batch_size = 32):
        super().__init__()
        self.batch_size = batch_size
        self.resnet = ResNet(pretrained=True)
        self.bilstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc = nn.Linear(512, 1)
    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        bilstm_out, bilstm_hc = self.bilstm(x)
        x = torch.cat([bilstm_hc[0][0], bilstm_hc[0][1]], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        cost = self.fc(x)
        #print('D(s): ', cost.item())
        return 0.4999*torch.tanh(cost) + 0.5

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet_model.conv1.weight = nn.Parameter(self.resnet_model.conv1.weight[:, 0:2]) 
        
    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        out1 = self.resnet_model.layer1(x)  # width, height=1/4
        out2 = self.resnet_model.layer2(out1)  # width, height=1/8
        out3 = self.resnet_model.layer3(out2)  # width, height=1/16
        out4 = self.resnet_model.layer4(out3)  # width, height=1/32
        out = self.resnet_model.avgpool(out4)
        #return out1, out2, out3, out4
        return out


if __name__ == '__main__':
    from torchvision import transforms as transforms
    from dataloader import DataLoder
    import numpy as np
    if torch.cuda.is_available():
         device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dataloader = DataLoder("mask_imgs", shape=256)
    data = dataloader.data
    model = ActorCriticNet().to(device)
    obs = np.zeros((len(data), 1, 256, 256), dtype=np.float32)
    print(len(data))
    for i in range(len(data)):
        obs[i] = data[i]
    obs = torch.tensor(obs, device=device)
    print(model(obs[0:8]))

