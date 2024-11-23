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
actions = [0,1,2,3,4]
agents = []

# # Q-learning parameters
# alpha = 1             # Learning rate
# gamma = 0.99             # Discount factor for future rewards
# epsilon = 1e-1           # Epsilon for epsilon-greedy policy
# threshold = 2e4          # Convergence threshold for episodic reward value

# env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range, 'linear')
# obs_ = env.reset()

# Function to calculate agent-specific rewards
def agent_reward(next_state, agent):
    goal_loc = (env.goal.x, env.goal.y) if agent == 3 else (env.plates[agent].x, env.plates[agent].y)
    curr_room = env._get_curr_room_reward(next_state[1])
    agent_loc = (next_state[0], next_state[1])
    
    # Reward calculation based on agent proximity to goal and room
    if agent == curr_room:
        reward = - np.linalg.norm((np.array(goal_loc) - np.array(agent_loc)), 1) / env.max_dist
    else:
        reward = -len(env.room_boundaries) + 1 + curr_room
    
    return reward


# Function to determine the next state based on the current state and action
def state_transition(state, action):
    stat = state.copy()
    proposed_pos = [state[0], state[1]]
    
    if action == 0:  # Move Left
        proposed_pos[1] -= 1
    elif action == 1:  # Move Right
        proposed_pos[1] += 1
    elif action == 2:  # Move Up
        proposed_pos[0] -= 1
    elif action == 3:  # Move Down
        proposed_pos[0] += 1
    # action == 4 is NOOP (no operation)

    # Update state only if there's no collision
    if not env._detect_collision(proposed_pos):
        stat = proposed_pos
    
    return stat

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
    state_dim = gridsize[0]*gridsize[1]
    
    action_dim = 5
    for i in range(n_agents):
        agents.append(DQNAgent(state_dim, action_dim))
    
    # print(agent)
    episodes = 500
    batch_size = 32
    
    for episode in range(episodes):
        state = env.reset()
        state = np.ravel(state)  # Flatten state
        total_reward = 0
        print(f"episode is {episode}-----------------------")
        for step in range(500):  # Maximum steps per episode
            for i in range(n_agents):
                action = agents[i].act(state)
                # print(agent)
                temp_action = [4]*n_agents
                
                # print(action)
                
                temp_action[i] =  action
                next_state, rd, done, _ = env.step(temp_action)
                # next_state = np.ravel(next_state)  # Flatten next state

                stat = np.array(next_state[i][a_co:a_co + 2], dtype=int)

                reward = agent_reward(stat,i)
                agents[i].store_transition(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                env.render()
                
                # if done:
                #     break
        
        agents.replay(batch_size)
        agents.update_target_model()
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")
