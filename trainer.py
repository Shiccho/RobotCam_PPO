import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, model, discriminator, t_num, env, data, resolution, optimizer, optimizer_d, updates, epochs, batchsize, mini_batchsize, lr, max_norm, norm_type, gamma, lamda, device):
        self.model = model
        self.discriminator = discriminator
        self.t_num = t_num
        self.env = env
        tau_coef = 3
        self.data = data
        self.data = data[tau_coef * t_num:]
        self.tau_E = self.data[0:tau_coef * t_num]
        tau_A_array = np.zeros((tau_coef * t_num, t_num, 1, 256, 256), dtype=np.float32)
        self.tau_A = torch.tensor(tau_A_array)
        for i in range(tau_coef*t_num):
            self.tau_A[i], _, __, ___, = env.step(np.random.randint(0, 12))
        self.obs = torch.tensor(np.zeros((1, t_num, 1, resolution[0], resolution[1]), dtype=np.float32))
        self.obs[0, :] = env.reset()
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.epochs = epochs
        self.batchsize = batchsize
        self.mini_batchsize = mini_batchsize
        self.lr = lr
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.resolution = resolution
        self.gamma = gamma
        self.lamda = lamda
        self.updates = updates
        self.device = device
        self.log = {}

    def _calcLoss(self, samples, data, clip_range):
        ##Agent Loss
        sampled_return = samples['values'] + samples['advantages']
        sampled_advantage = samples['advantages']
        sampled_value = samples['values']
        log_pi = torch.tensor(np.zeros(samples['log_pis'].shape, dtype=np.float32))
        value = torch.tensor(np.zeros(sampled_value.shape, dtype=np.float32))
        for i in range(self.mini_batchsize):
            pi, value_gpu = self.model(samples['obs'][i].to(self.device))
            #print('value', value[0].item(),'\t', 'action', pi.sample()[0][0].item())
            log_pi[i] = pi.log_prob(samples['actions'][i].to(self.device)).cpu()
            value[i] = value_gpu.cpu()
        log_pi_old = samples['log_pis']
        ratio = torch.exp(log_pi - log_pi_old)
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_advantage, 
                                 clipped_ratio * sampled_advantage)
        policy_reward = policy_reward.mean()

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()
        val_clip_range = clip_range
        clippedValue = sampled_value + (value - sampled_value).clamp(min = -val_clip_range, max = val_clip_range)
        Lvf = torch.max((value - sampled_return)**2, (clippedValue - sampled_return)**2)
        Lvf = Lvf.mean()

        c1 = 5e-1
        c2 = 1e-1
        loss = -(policy_reward - c1*Lvf + c2*entropy_bonus)

        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()
        self.log['policy_reward'] = policy_reward.item()
        self.log['vf_loss'] = Lvf.item()
        self.log['entropy_bonus'] = entropy_bonus.item()
        self.log['clip_fraction'] = clip_fraction.item()

        ##Discriminator Loss
        d_pi_A = torch.tensor(np.zeros((self.mini_batchsize, 1), dtype=np.float32))
        d_pi_E = torch.tensor(np.zeros((self.mini_batchsize, 1), dtype=np.float32))
        for i in range(self.mini_batchsize):
            d_pi_A[i] = self.discriminator(samples['obs'][i].to(self.device))
            d_pi_E[i] = self.discriminator(data[i].to(self.device))
        log_d_pi = torch.log(1.-d_pi_A)
        log_d_pi_E = torch.log(d_pi_E)
        loss_d = - log_d_pi.mean() - log_d_pi_E.mean()

        indexes = torch.randperm(self.tau_E.shape[0])
        start = np.random.randint(0, indexes.shape[0]-self.t_num)
        end = start + self.t_num
        d_tau_A = torch.tensor(np.zeros((self.mini_batchsize, 1), dtype=np.float32))
        d_tau_E = torch.tensor(np.zeros((self.mini_batchsize, 1), dtype=np.float32))
        for i in range(self.mini_batchsize):
            tau_E_state_i = self.tau_E[indexes[start:end]]
            tau_A_state_i = self.tau_A[np.random.randint(0, self.tau_A.shape[0])]
            d_tau_E[i] = self.discriminator(tau_E_state_i.to(self.device))
            d_tau_A[i] = self.discriminator(tau_A_state_i.to(self.device))
        log_d_tau_E = torch.log(d_tau_E)
        log_d_tau_A = torch.log(1.-d_tau_A)
        loss_d_tau = - log_d_tau_E.mean() - log_d_tau_A.mean()
        acc_E = torch.where(d_tau_E > 0.5, 1., 0.).mean()
        acc_A = torch.where(d_tau_A < 0.5, 1., 0.).mean()
        acc = (acc_E + acc_A)
        if acc.item() > 1.0:
            loss_d = loss_d - loss_d_tau

        self.log['loss_d'] = loss_d.item()
        self.log['d_pi'] = d_pi_A.mean().item()
        self.log['d_pi_E'] = d_pi_E.mean().item()
        self.log['acc_E'] = acc_E.item()
        self.log['acc_A'] = acc_A.item()
        self.log['acc'] = acc.item() / 2.
        
        return loss, loss_d

    def _calcAdvantages(self, done, rewards, values):
        advantages = np.zeros((1, self.batchsize), dtype=np.float32)
        lastAdvantage = 0

        _, lastValue = self.model(self.obs[0].to(self.device))
        lastValue = lastValue.cpu().data.numpy()

        for t in reversed(range(self.batchsize)):
            mask = 1.0 - done[:, t]
            lastValue = lastValue * mask
            lastAdvantage = lastAdvantage * mask
            delta = rewards[:, t] + self.gamma * lastValue - values[:, t]
            lastAdvantage = delta + self.gamma * self.lamda * lastAdvantage
            advantages[:, t] = lastAdvantage
            lastValue = values[:, t]
        
        return advantages
    
    def _normalize(self, tensor: torch.Tensor):
        normalized_tensor = tensor.clamp(max=1., min=-1.)
        return normalized_tensor

    def sample(self):
        rewards = np.zeros((1, self.batchsize), dtype = np.float32)
        actions = np.zeros((1, self.batchsize), dtype=np.int32)
        done = np.zeros((1, self.batchsize), dtype=np.bool)
        obs = np.zeros((1, self.batchsize, self.t_num, 1, self.resolution[1], self.resolution[0]), dtype=np.float32)
        log_pis = np.zeros((1, self.batchsize), dtype=np.float32)
        values = np.zeros((1, self.batchsize), dtype=np.float32)

        for t in range(self.batchsize):
            with torch.no_grad():
                obs[:, t] = self.obs.cpu().numpy()
                pi, v = self.model(self.obs[0].to(self.device))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()
        
            self.obs[0,:], _, done[0, t], info = self.env.step(a)
            with torch.no_grad():
                #rewards[0, t] = self._normalize(-torch.log(1.-self.discriminator(self.obs))).cpu().numpy()
                rewards[0, t] = -torch.log(1.-self.discriminator(self.obs[0].to(self.device))).cpu().numpy()
            if done[0, t] == True:
                self.obs[0, :] = self.env.reset()
                self.log['reward'] = rewards.sum()
        advantages = self._calcAdvantages(done, rewards, values)
        samples = {
            'obs' : obs,
            'actions' : actions,
            'values' : values,
            'log_pis' : log_pis,
            'advantages' : advantages
        }
        
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0]*v.shape[1], *v.shape[2:])
            samples_flat[k] = torch.tensor(v, dtype=torch.float32)

        return samples_flat
        

    def train(self, samples, data, clip_range):
        for _ in range(self.epochs):
            indexes = torch.randperm(self.batchsize)
            for start in range(0, self.batchsize, self.mini_batchsize):
                end = start + self.mini_batchsize
                minibatch_indexes = indexes[start: end]
                minibatch = {}
                minibatch_data_indexes = indexes % (data.shape[0]-self.t_num)
                minibatch_data = torch.tensor(np.zeros((self.mini_batchsize, self.t_num, 1, self.resolution[1], self.resolution[0]), dtype=np.float32))
                for k, v in samples.items():
                    minibatch[k] = v[minibatch_indexes]
                for i in range(self.mini_batchsize):
                    minibatch_data[i, :] = data[minibatch_data_indexes[i] : minibatch_data_indexes[i]+self.t_num]
                loss, loss_d = self._calcLoss(clip_range = clip_range, samples=minibatch, data=minibatch_data)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.lr
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type)
                self.optimizer.step()

                for pg in self.optimizer_d.param_groups:
                    pg['lr'] = self.lr
                self.optimizer_d.zero_grad()
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.max_norm, norm_type=self.norm_type)
                self.optimizer_d.step()

    def runTrainingLoop(self):
        writer = SummaryWriter(log_dir="logs")
        for update in range(0, self.updates):
            progress = update / self.updates
            self.lr = 2.5e-4 * (1-(1.0/2.5)*progress)
            clip_range = 0.2 * (1-progress)

            samples = self.sample()
            self.train(samples, self.data, clip_range)
            
            if (update + 1) % 1 == 0:
                print('===========================================================================================================')
                print("Progress:{0}/{1}\t|Loss_A:{2}\t Reward:{3:.1f}\t Loss_D:{4}\n \
                       -----------------------------------------------------------------------------------\n \
                       |PolicyReward:{5:.3f}\tVF_Loss:{6:.3f}\tEntropyBounus:{7:.3f}\tClipFraction:{8:.2f}\n \
                       |D_pi_A:{9:.2f}\tD_pi_E:{10:.2f}\tAcc.:{11:.2f}(A:{12:.2f} E:{13:.2f})".format(
                       update, self.updates,
                       -(self.log['policy_reward'] - 5e-1 * self.log['vf_loss'] + 1e-1 * self.log['entropy_bonus']),
                       self.log['reward'],
                       self.log['loss_d'],
                       self.log['policy_reward'],
                       self.log['vf_loss']*5e-1, 
                       self.log['entropy_bonus']*1e-1,
                       self.log['clip_fraction'],
                       self.log['d_pi'],
                       self.log['d_pi_E'],
                       self.log['acc'],
                       self.log['acc_A'],
                       self.log['acc_E']
                       ))
                print('===========================================================================================================')
                writer.add_scalar('Agent/Lclip', self.log['policy_reward'], global_step=update)
                writer.add_scalar('Agent/Lv', self.log['vf_loss'], global_step=update)
                writer.add_scalar('Agent/Sf(pi)', self.log['entropy_bonus'], global_step=update)
                writer.add_scalar('Discriminator/Loss', self.log['loss_d'], global_step=update)
            
            if (update + 1) % 100 == 0:
                model_path = 'model.pth'
                torch.save(self.model.state_dict(), model_path)
                #self.test()
                #writer.add_scalar('Success Rate', self.testLog['success']/self.numWorkers, global_step=update)