import torch
from torch import optim
from dataloader import DataLoder
from environment import Env
from trainer import Trainer
from model import ActorCriticNet, Discriminator
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class Main():
    def __init__(self):
        updates = 100000
        epochs = 1
        batchsize = 128
        num_minibatch = 8
        mini_batchsize = batchsize // num_minibatch
        max_norm = 0.5
        norm_type = 2.
        gamma = 0.99
        lamda = 0.95

        t_num = 8
        model = ActorCriticNet(batch_size=t_num).to(device)
        discriminator = Discriminator(batch_size=t_num).to(device)
        resolution = (256, 256)
        env = Env(t_num)

        lr = 2.5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
        
        self.dataloader = DataLoder('mask_imgs', shape=256)
        data = np.zeros((len(self.dataloader.data), 1, 256, 256), dtype=np.float32)
        for i in range(len(data)):
            data[i] = self.dataloader.data[i]
        data = torch.tensor(data)


        self.trainer = Trainer(model, discriminator, t_num,
                               env, data, resolution,
                               optimizer,
                               optimizer_d,
                               updates,
                               epochs, batchsize, mini_batchsize, 
                               lr,
                               max_norm, norm_type,
                               gamma,
                               lamda,
                               device)
        
        
    
    def main(self):
        self.trainer.runTrainingLoop()

if __name__ == '__main__':
    main = Main()
    main.main()
