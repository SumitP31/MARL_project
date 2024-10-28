from environment import PressurePlate
import gym
import time
import numpy as np
import random

# Hyperparameters
n_episodes = 50          # Total number of episodes to run
max_steps = 1000         # Max steps per episode
gridsize = (15, 9)       # Size of the grid environment
n_agents = 4             # Number of agents in the environment
n_actions = 5            # Number of possible actions each agent can take
a_range = 2              # Agent's observation range
a_co = ((2 * a_range + 1) ** 2) * 4  # Index for agent coordinates in observation

# Q-learning parameters
alpha = 1             # Learning rate
gamma = 0.99             # Discount factor for future rewards
epsilon = 1e-1           # Epsilon for epsilon-greedy policy
threshold = 2e4          # Convergence threshold for episodic reward value

# Initialize environment and Q-values
env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range, 'linear')

obs_ = env.reset()

Q = np.zeros((n_agents, gridsize[1], gridsize[0], n_actions), dtype=float)

policy = np.zeros((n_agents, gridsize[1], gridsize[0]), dtype=int)
#--------------------------------------------------------------------------------------------------------


# Function to evaluate current policy by selecting actions for each agent based on the state
def policy_eval(state):
    actions = []
    for i in range(n_agents):
        x, y = state[i][0], state[i][1]
        actions.append(policy[i, x, y])
    return actions


# Q-value update and action selection function
def q_value(Q, state, agent):
    for a in range(n_actions):
        st = state.copy()
        next_st = state_transition(st, a)
        reward = agent_reward(next_st, agent)
        
        # Update Q-value using the Bellman equation
        Q[st[0], st[1], a] += alpha * (reward + gamma * np.max(Q[next_st[0], next_st[1]]) - Q[st[0], st[1], a])
    
    action = np.argmax(Q[state[0], state[1]])
    return action, Q


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


# Main function to run the simulation
def main():
    for episode in range(n_episodes):
        obs_ = env.reset()
        state = np.array([obs_[i][a_co:a_co + 2] for i in range(n_agents)], dtype=int)
        
        for step in range(max_steps):
            for agent in range(n_agents):
                temp_act = [4] * n_agents  # Initialize actions as NOOP
                action, q = q_value(Q[agent], state[agent], agent)
                
                temp_act[agent] = action
                obs, rewards, done, _ = env.step(temp_act)
                
                # Update policy with selected action
                policy[agent, state[agent][0], state[agent][1]] = action
                state[agent] = np.array(obs[agent][a_co:a_co + 2], dtype=int)
                
                if all(done):
                    print(f"Episode: {episode} finished")
                    break
                
                env.render()
                # Uncomment to slow down rendering
                # time.sleep(0.1)

if __name__ == "__main__":
    main()
