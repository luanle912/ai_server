import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

_job_cols_ = 3
_window_size_ = 50
history_job_dict = {}

class A2C(nn.Module):
    def __init__(self, num_inputs, num_outputs, std=0.0, window_size=50,
                 learning_rate=1e-2, gamma=0.99, batch_size=20, layer_size=[]):
        super(A2C, self).__init__()
        self.hidden1_size = layer_size[0]
        self.hidden2_size = layer_size[1]
        self.critic = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, self.hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden1_size, self.hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden2_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Conv1d(2, 1, 1),
            nn.Flatten(start_dim=0),
            nn.Linear(num_inputs, self.hidden1_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden1_size, self.hidden2_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden2_size, num_outputs)
        )
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = learning_rate
        self.window_size = window_size
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.rewards_seq = []
        self.entropy = 0

    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 1))
        value = self.critic(x)
        probs = self.actor(x)
        return probs, value

    def remember(self, probs, value, reward, done, device, action):
        dist = Categorical(torch.softmax(probs, dim=-1))
        log_prob = dist.log_prob(torch.tensor(action))
        self.entropy += dist.entropy().mean()

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.FloatTensor(
            [reward]).unsqueeze(-1).to(device))
        self.rewards_seq.append(reward)

    def train(self, next_value, optimizer):
        if len(self.values) < self.batch_size:
            return

        returns = self.compute_returns(next_value)

        self.log_probs = torch.tensor(self.log_probs)
        returns = torch.cat(returns).detach()
        self.values = torch.cat(self.values)

        advantage = returns - self.values

        actor_loss = -(self.log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropy = 0

    def compute_returns(self, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step][0] + self.gamma * R
            returns.insert(0, R)
        return returns

    def save_using_model_name(self, model_name_path):
        torch.save(self.state_dict(), model_name_path + ".pkl")

    def load_using_model_name(self, model_name_path='model/agent'):
        self.load_state_dict(
            torch.load(model_name_path + ".pkl"))

def preprocessing_queued_jobs(wait_job, currentTime):
    job_info_list = []
    for job in wait_job:
        s = float(job['submissionTime'])
        t = float(job['userEst'])
        t_new = get_reqTime_from_history(job)
        print('t: ', t, 't_new: ', t_new, 'is_first: ', t == t_new)
        n = float(job['procReq'])
        w = int(currentTime - s)
        i = int(job['userId'])
        e = int(job['exe_num'])
        # award 1: high priority; 0: low priority
        # a = int(wait_job[i]['award'])
        info = [[n, t], [0, w]]
        if _job_cols_ == 3:
            info = [[n, t_new], [0, w], [i, e]]
        # info = [[n, t], [a, w]]
        job_info_list.append(info)
    return job_info_list

def preprocessing_system_status(node_struc, currentTime):
    node_info_list = []
    # Each element format - [Availbility, time to be available] [1, 0] - Node is available
    for node in node_struc:
        info = []
        # avabile 1, not available 0
        if node['state'] < 0:
            info.append(1)
            info.append(0)
        else:
            info.append(0)
            info.append(node['end'] - currentTime)
            # Next available node time.

        node_info_list.append(info)
    return node_info_list

def make_feature_vector(jobs, system_status):
    # Remove hard coded part !
    job_cols = _job_cols_
    window_size = _window_size_
    input_dim = [len(system_status) + window_size *
                    job_cols, len(system_status[0])]
    fv = np.zeros((1, input_dim[0], input_dim[1]))
    i = 0
    for idx, job in enumerate(jobs):
        fv[0, idx * job_cols:(idx + 1) * job_cols, :] = job
        i += 1
        if i == window_size:
            break
    fv[0, job_cols * window_size:, :] = system_status
    return fv


def get_reqTime_from_history(job):
    if job['userId'] not in history_job_dict:
        return job['userEst']
    else:
        print(history_job_dict[job['userId']])
        return job['userEst'] / (np.mean([x['ratio'] for x in history_job_dict[job['userId']]]) + 1e-6)
    
def get_action_from_output_vector(output_vector, wait_queue_size, is_training):
    action_p = torch.softmax(
        output_vector[:wait_queue_size], dim=-1)
    action_p = np.array(action_p)
    action_p /= action_p.sum()
    if is_training:
        wait_queue_ind = np.random.choice(len(action_p), p=action_p)
    else:
        wait_queue_ind = np.argmax(action_p)
    return wait_queue_ind