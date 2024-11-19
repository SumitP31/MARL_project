from environment import PressurePlate
import gym
import time
import numpy as np
import random

# Hyperparameters
n_episodes = 100          # Total number of episodes to run
max_steps = 500         # Max steps per episode
gridsize = (28, 16)       # Size of the grid environment
n_agents = 4             # Number of agents in the environment
n_actions = 5            # Number of possible actions each agent can take
a_range = 1              # Agent's observation range
a_co = ((2 * a_range + 1) ** 2) * 4  # Index for agent coordinates in observation
plate_id = [((2 * a_range + 1) ** 2) * 3, (((2 * a_range + 1) ** 2) * 4)-1]
global plate_found
plate_found = [False] * n_agents
# Q-learning parameters
alpha = 1     # Learning rate
gamma = 1     # Discount factor for future rewards
epsilon = 1e-3    # Epsilon for epsilon-greedy policy
threshold = 2e4  # Convergence threshold for episodic reward value

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

# Function to calculate agent-specific rewards
def agent_reward(next_state, agent):
    
    goal_loc = (env.goal.x, env.goal.y) if agent == (n_agents-1) else (env.plates[agent].x, env.plates[agent].y)
    curr_room = env._get_curr_room_reward(next_state[agent][1])
    agent_loc = (next_state[agent][0], next_state[agent][1])
    
    
    obs = env._get_obs(True, next_state, agent)
    plt_obs =   obs[agent][plate_id[0] : plate_id[1]+1] #np.array([ for i in range(n_agents)])
    plate_found[agent] = 1 in plt_obs
    # print(f"Observe this : {plate_found}\n end of observation")
    # Reward calculation based on agent proximity to goal and room
    if agent == curr_room:
        if plate_found[agent] == True:
            reward = - np.linalg.norm((np.array(goal_loc) - np.array(agent_loc)), 1) / env.max_dist
        else:
            reward = float(-3.0)
    else:
        reward = -len(env.room_boundaries) + 1 + curr_room
        
    # print(f"for agent{agent} the reward is {reward} if plate found is {plate_found}")
    return reward

def platefound():
    f=1

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


# Q-value update and action selection function
def q_value(Q, state, agent):
    for a in range(n_actions):
        st = state.copy()
        st_ = state.copy()
        next_st = state_transition(st[agent], a)
        st_[agent] = next_st       
        reward = agent_reward(st_, agent)
        # print(f"reward for agent {agent} at state {st_} {reward}")
        # Update Q-value using the Bellman equation
        Q[st[agent][0], st[agent][1], a] += alpha * (reward + gamma * np.max(Q[next_st[0], next_st[1]]) - Q[st[agent][0], st[agent][1], a])
    
    action = np.argmax(Q[state[agent][0], state[agent][1]])
    return action, Q


# Main function to run the simulation
def main():
    for episode in range(n_episodes):
        obs_ = env.reset()
        state = np.array([obs_[i][a_co:a_co + 2] for i in range(n_agents)], dtype=int)
        
        plate_found = [False] * n_agents 
        
        for step in range(max_steps):
            for agent in range(n_agents):
                temp_act = [4] * n_agents  # Initialize actions as NOOP
                action, q = q_value(Q[agent], state, agent)
                
                temp_act[agent] = action
                obs, rewards, done, _ = env.step(temp_act)
                
                # Update policy with selected action
                policy[agent, state[agent][0], state[agent][1]] = action
                state[agent] = np.array(obs[agent][a_co:a_co + 2], dtype=int)
                
                if all(done):
                    # print(f"Episode: {episode} finished")
                    break
                
                # if episode > n_episodes-10:
                env.render()
                # Uncomment to slow down rendering
                # time.sleep(0.05)
        print(f"Episode: {episode}")
        
if __name__ == "__main__":
    main()
