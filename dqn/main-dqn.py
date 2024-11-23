import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from environment import PressurePlate

# Hyperparameters
n_episodes = 2          # Total number of episodes to run
max_steps = 1000         # Max steps per episode
gridsize = (15, 9)       # Size of the grid environment
n_agents = 4             # Number of agents in the environment
n_actions = 5            # Number of possible actions each agent can take
a_range = 2              # Agent's observation range
a_co = ((2 * a_range + 1) ** 2) * 4  # Index for agent coordinates in observation

# # Q-learning parameters
# alpha = 1             # Learning rate
# gamma = 0.99             # Discount factor for future rewards
# epsilon = 1e-1           # Epsilon for epsilon-greedy policy
# threshold = 2e4          # Convergence threshold for episodic reward value

# env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range, 'linear')
# obs_ = env.reset()

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the DQN
if __name__ == "__main__":
    env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range, 'linear')
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    batch_size = 32
    
    for episode in range(episodes):
        state = env.reset()
        state = np.ravel(state)  # Flatten state
        total_reward = 0
        
        for step in range(500):  # Maximum steps per episode
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.ravel(next_state)  # Flatten next state
            
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay(batch_size)
        agent.update_target_model()
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")