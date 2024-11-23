import gym
from gym import spaces
import numpy as np

class PressurePlate(gym.Env):
    def __init__(self, width, height, n_agents, a_range, reward_type):
        super().__init__()
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.a_range = a_range
        self.reward_type = reward_type

        # Observation space remains unchanged
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_agents, 2 * a_range + 1, 2 * a_range + 1), dtype=np.float32
        )

        # Flatten the action space for compatibility with single DQN
        self.original_action_space = spaces.Tuple([spaces.Discrete(5)] * n_agents)
        self.action_space = spaces.Discrete(5**n_agents)

        # Internal state
        self.agents = [(np.random.randint(0, self.width), np.random.randint(0, self.height)) for _ in range(n_agents)]
        self.goal = (self.width - 1, self.height - 1)
        self.step_count = 0
        self.max_steps = 500

    def reset(self):
        self.agents = [(np.random.randint(0, self.width), np.random.randint(0, self.height)) for _ in range(self.n_agents)]
        self.step_count = 0
        obs = self._get_observations()
        return obs

    def step(self, action):
        # Map the flattened action back to individual agent actions
        actions = self._unflatten_action(action)
        
        rewards = 0
        done = False
        
        # Process actions for all agents
        for idx, act in enumerate(actions):
            self.agents[idx] = self._take_action(self.agents[idx], act)

        # Calculate rewards (this remains unchanged)
        if self.reward_type == "linear":
            rewards = self._calculate_rewards()
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_observations()
        return obs, rewards, done, {}

    def _take_action(self, agent_pos, action):
        # This function remains unchanged
        if action == 0:  # Up
            new_pos = (agent_pos[0], max(0, agent_pos[1] - 1))
        elif action == 1:  # Down
            new_pos = (agent_pos[0], min(self.height - 1, agent_pos[1] + 1))
        elif action == 2:  # Left
            new_pos = (max(0, agent_pos[0] - 1), agent_pos[1])
        elif action == 3:  # Right
            new_pos = (min(self.width - 1, agent_pos[0] + 1), agent_pos[1])
        else:  # Stay
            new_pos = agent_pos
        return new_pos

    def _get_observations(self):
        # This function remains unchanged
        observations = []
        for agent_pos in self.agents:
            obs = np.zeros((2 * self.a_range + 1, 2 * self.a_range + 1), dtype=np.float32)
            for x in range(-self.a_range, self.a_range + 1):
                for y in range(-self.a_range, self.a_range + 1):
                    grid_x = agent_pos[0] + x
                    grid_y = agent_pos[1] + y
                    if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                        obs[x + self.a_range, y + self.a_range] = 1
            observations.append(obs)
        return np.array(observations)

    def _calculate_rewards(self):
        # This function remains unchanged
        rewards = 0
        for agent_pos in self.agents:
            if agent_pos == self.goal:
                rewards += 10
        return rewards

    def _unflatten_action(self, action):
        """Converts a single flattened action into individual agent actions."""
        actions = []
        for _ in range(self.n_agents):
            actions.append(action % 5)
            action //= 5
        return actions
