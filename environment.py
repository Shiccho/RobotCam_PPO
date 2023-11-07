import gym
import gym_unrealcv
import torch
import numpy as np

class Env(gym.Env):
    def __init__(self, t_num, num_agent, transform):
        self.env = gym.make('B4Thesis-Multi-v0')
        self.agent = num_agent
        self.state = torch.tensor(np.zeros((self.agent, t_num, 2, 64, 64), dtype=np.float32))
        self.transform = transform
    
    def step(self, actions):
        obs, rewards, done, info = self.env.step(actions)
        for a in range(self.agent):
            self.state[a, :, 0] = self.enroll_state(self.state[a, :, 0], self.transform(obs[a][0]))
            self.state[a, :, 1] = self.enroll_state(self.state[a, :, 1], self.transform(obs[a][1]))
        return self.state, rewards, done, info
    
    def reset(self):
        obs = self.env.reset()
        for a in range(self.agent):
            self.state[a, :, 0] = self.transform(obs[a][0])
            self.state[a, :, 1] = self.transform(obs[a][1])
        return self.state

    def test_reset(self):
        obs = self.env.test_reset()
        for a in range(self.agent):
            self.state[a, :, 0] = self.transform(obs[a][0])
            self.state[a, :, 1] = self.transform(obs[a][1])
        return self.state

    
    def enroll_state(self, state, obs):
        state = np.roll(state.cpu().numpy(), shift=-1, axis=0)
        state[-1] = obs.cpu().numpy()
        state = torch.tensor(state)
        return state

if __name__ == "__main__":
    env = Env(8)
    print(env.reset().shape)

