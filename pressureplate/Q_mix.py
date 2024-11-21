import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QMixer(nn.Module):
    """
    QMIX Network that combines individual agent Q-values 
    using a hypernet with monotonic constraints
    """
    def __init__(self, n_agents, state_dim, mixing_embed_dim=32):
        super(QMixer, self).__init__()
        
        # Hypernet to generate weights and biases
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim * n_agents)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )
        
        # Bias terms
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
        
        self.n_agents = n_agents

    def forward(self, indiv_qs, states):
        """
        Combine individual agent Q-values
        
        Args:
        - indiv_qs: Individual Q-values for each agent [batch_size, n_agents]
        - states: Global state [batch_size, state_dim]
        
        Returns:
        - Combined Q-value [batch_size]
        """
        # Generate first layer weights 
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.n_agents, w1.size(-1) // self.n_agents)
        
        # Generate first layer bias
        b1 = self.hyper_b1(states)
        
        # First mixing layer
        hidden = F.elu(torch.matmul(indiv_qs.unsqueeze(1), w1) + b1)
        
        # Generate second layer weights
        w2 = torch.abs(self.hyper_w2(states))
        
        # Generate second layer bias
        b2 = self.hyper_b2(states)
        
        # Final mixing layer
        q_tot = torch.matmul(hidden, w2.unsqueeze(-1)) + b2
        
        return q_tot.squeeze(-1)

class IndividualQNetwork(nn.Module):
    """
    Q-Network for individual agents
    """
    def __init__(self, obs_dim, action_dim):
        super(IndividualQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QMIXTrainer:
    """
    QMIX Training Framework for Multi-Agent Reinforcement Learning
    """
    def __init__(self, n_agents, obs_dim, action_dim, state_dim):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Individual Q-Networks for each agent
        self.agent_networks = [
            IndividualQNetwork(obs_dim, action_dim) 
            for _ in range(n_agents)
        ]
        
        # QMIX mixer network
        self.mixer = QMixer(n_agents, state_dim)
        
        # Optimizer
        self.mixer_optimizer = torch.optim.Adam(self.mixer.parameters(), lr=1e-4)
        self.agent_optimizers = [
            torch.optim.Adam(agent.parameters(), lr=1e-4) 
            for agent in self.agent_networks
        ]
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
    def select_actions(self, observations):
        """
        Select actions for all agents using epsilon-greedy
        
        Args:
        - observations: List of agent observations
        
        Returns:
        - List of selected actions
        """
        actions = []
        for i, obs in enumerate(observations):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs)
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                # Random action
                action = np.random.randint(0, self.action_dim)
            else:
                # Greedy action
                with torch.no_grad():
                    q_values = self.agent_networks[i](obs_tensor)
                    action = q_values.argmax().item()
            
            actions.append(action)
        
        return actions
    
    def train(self, batch):
        """
        Train QMIX algorithm
        
        Args:
        - batch: Training batch containing:
          - observations
          - actions
          - rewards
          - next_observations
          - states
          - next_states
          - dones
        """
        # Unpack batch
        obs, actions, rewards, next_obs, states, next_states, dones = batch
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_obs_tensor = torch.FloatTensor(next_obs)
        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Compute individual Q-values
        curr_agent_qs = torch.stack([
            agent(obs) for agent, obs in zip(self.agent_networks, obs_tensor)
        ], dim=1)
        
        # Select Q-values for taken actions
        curr_agent_qs = curr_agent_qs.gather(
            -1, actions_tensor.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute target Q-values
        with torch.no_grad():
            # Next step Q-values for each agent
            next_agent_qs = torch.stack([
                agent(next_obs) for agent, next_obs in 
                zip(self.agent_networks, next_obs_tensor)
            ], dim=1)
            
            # Max Q-values for next actions
            next_agent_qs, _ = next_agent_qs.max(dim=-1)
            
            # Compute total Q-value using mixer
            curr_mixer_q = self.mixer(curr_agent_qs, states_tensor)
            next_mixer_q = self.mixer(next_agent_qs, next_states_tensor)
            
            # Compute target
            target_q = rewards_tensor + self.gamma * next_mixer_q * (1 - dones_tensor)
        
        # Compute loss
        mixer_loss = F.mse_loss(curr_mixer_q, target_q)
        
        # Backpropagate mixer loss
        self.mixer_optimizer.zero_grad()
        mixer_loss.backward()
        self.mixer_optimizer.step()
        
        # Compute and backpropagate individual agent losses
        for i, agent in enumerate(self.agent_networks):
            agent_loss = F.mse_loss(curr_agent_qs[:, i], target_q)
            
            self.agent_optimizers[i].zero_grad()
            agent_loss.backward()
            self.agent_optimizers[i].step()
        
        return mixer_loss.item()

# Example usage would involve creating a multi-agent environment 
# and using this framework to train agents