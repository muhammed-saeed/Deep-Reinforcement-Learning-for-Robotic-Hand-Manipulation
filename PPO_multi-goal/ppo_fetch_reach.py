import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        #in our case self.inputs ==> input state//goal
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        #in ddpg you can learn from data stored from older epochs
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    '''
    continous action space PG  methods the actor returns tensor of lenght number of action
    and those tensors are  mean and covariance -- use Gaussian Distribution to Sample from action space
    covarinace is used to balance exploration-exploitation problem 
    '''
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()
                )
                #the output of network is the mean of actions
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
                )
        #torch.full   returns torch_tensor of size action_dim and filled with value action_std*action_std
        #action_var --> variance of the action
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        #
        action_mean = self.actor(state)
        #the covariance is 2D diagonal array of action variance
        #torch.diagonal --> since action_var is one_dim tensor torch.diag returns 2D array with tensor's values as main diagonal
        cov_mat = torch.diag(self.action_var).to(device)
        #sample actions from Gaussian Distribution with mean = action_mean, and covariance = action variance, cov_mat
        dist = MultivariateNormal(action_mean, cov_mat)
        #sample action
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        #torch.diag_embed returns 2D diagnoal array with tensor's elements as main diagonal
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        # its probablitis not values Pi(a|s)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        #state.reshape --> change state into vector - single row - of compitable length with the orignal state
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        #loop through all episodes in the epoch
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            #insert the new discounted reward - after completing the episode - in index number 0 and push old ones
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            #exp(log(prob)) = prob
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            #total_loss_function = l_clip, loss_critic, loss_entropy
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

epoch_success_rate = []        

clip_obs = 100
def _preproc_og( o, g):
        #(o - o.mean())/o.std()
        o = np.clip(o, -clip_obs, clip_obs)
        #(g-g.mean())/g.std()
        g = np.clip(g, -clip_obs, clip_obs)
        return o, g

def launch():
    ############## Hyperparameters ##############
    env_name = "HandManipulateEggFull-v0"
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 800        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode
    
    update_timestep = 400      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 40               # update policy for K epochs
    eps_clip = 0.24              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.00101                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = 123
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim_pre = env.observation_space['observation'].shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]
    state_dim = state_dim_pre + goal_dim
    action_dim = env.action_space.shape[0]
    
    env.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    '''
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    '''
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    
    for epoch in range(350):
        running_reward = 0
        avg_length = 0
        time_step = 0
        
        for i_episode in range(1, max_episodes+1):
            episode_success_rate = []
            state_ = env.reset()
            env.env.reward_type = 'dense'

            obs = state_['observation']
            g = state_['desired_goal']

            #clip the oberservation and goal into range -200, 200
            #obs, g = _preproc_og(obs, g)
            state = np.concatenate([obs,g])

            local_timestep = 0
            for t in range(env._max_episode_steps):
                local_timestep += 1
                time_step +=1
                # Running policy_old:
                action = ppo.select_action(state, memory)
                state_, reward, done, info = env.step(action)
                #episode_success_rate.append(info['is_success'])
                #env.render()
                obs = state_['observation']
                g = state_['desired_goal']
                #obs, g = _preproc_og(obs, g)

                state = np.concatenate([obs,g])
                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                # update if its time
                #to go below
                
                running_reward += reward
                if done :
                    break

            if local_timestep<49:
                    print('Goaaaaaaaaaaaaaaaal')
            #episode_success_rate = np.array(episode_success_rate)
            #episode_success_rate = np.mean(episode_success_rate)
            
            
            avg_length += t
            # to go up form here ------>
            # stop training if avg_reward > solved_reward
            if running_reward > (log_interval*solved_reward):
                
                torch.save(ppo.policy.state_dict(), '/home/muhyahiarl/ppo_grad_project/PPO_continuous_{}.pth'.format(env_name))
                print("########## Solved! ##########")
                break
            
            # save every 400 episodes
            if i_episode % 400 == 0:
                torch.save(ppo.policy.state_dict(), '/home/muhyahiarl/ppo_grad_project/PPO_continuous_{}.pth'.format(env_name))
                print('updated')
            # logging
            '''
            if i_episode % log_interval == 0:
                avg_length = int(avg_length/log_interval)
                running_reward = int((running_reward/log_interval))
                
                print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0
            '''
            # unitl here <----
            # 
        # bring that bitch below to here |||||  
        # 
        #     
        # 
        
        
        print('reach here_0')
        ppo.update(memory)
        print('reach here_1')
        #memory.clear_memory()
        time_step = 0
        state_ = env.reset()
        env.env.reward_type = 'dense'
        print('reach here_2')
        obs = state_['observation']
        g = state_['desired_goal']
        state = np.concatenate([obs,g])
        local_timestep = 0

        test_success_rate = []
        for _ in range(10):
            local_success_rate = []
            state_ = env.reset()
            state = np.concatenate([state_['observation'], state_['desired_goal']])
            for t in range(env._max_episode_steps):
                local_timestep += 1
                time_step +=1
                # Running policy_old:
                action = ppo.select_action(state, memory)
                state_, reward, done, info = env.step(action)
                obs = state_['observation']
                g = state_['desired_goal']
                #obs, g = _preproc_og(obs, g)
                state = np.concatenate([obs ,g])
                local_success_rate.append(info['is_success'])
                if done:
                    break
            local_success_rate = np.array(local_success_rate)
            test_success_rate.append(np.mean(local_success_rate))

        local_test_success = np.mean(np.array(test_success_rate))
        epoch_success_rate.append(local_test_success)
        
        
        print('epoch ' +str(epoch) +' success rate is ',local_test_success)
        memory.clear_memory()
        avg_length += t 


def _eval_agent():
        test_success_rate = []
        for _ in range(10):
            local_success_rate = []
            state_ = env.reset()
            state = np.concatenate([state_['observation'], state_['desired_goal']])
            for t in range(env._max_episode_steps):
                local_timestep += 1
                time_step +=1
                # Running policy_old:
                action = ppo.select_action(state, memory)
                state_, reward, done, info = env.step(action)
                obs = state_['observation']
                g = state_['desired_goal']
                #obs, g = _preproc_og(obs, g)
                state = np.concatenate([obs ,g])
                local_success_rate.append(info['is_success'])
                if done:
                    break
            local_success_rate = np.array(local_success_rate)
            test_success_rate.append(np.mean(local_success_rate))

        return test_success_rate



if __name__ == '__main__':
    launch()
    np.savetxt('/home/muhyahiarl/ppo_grad_project/ppo_grad_project_handmanipulateeggfull_nepoch_350.txt',epoch_success_rate,delimiter=',')
