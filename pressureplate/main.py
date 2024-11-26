from environment import PressurePlate
import gym
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import randomcolor

# Hyperparameters
n_episodes = 1000       # Total number of episodes to run
max_steps = 1000         # Max steps per episode
gridsize = (28,16)      # Size of the grid environment (28,16) for max
n_agents = 4             # Number of agents in the environment
n_actions = 5            # Number of possible actions each agent can take
a_range = 1              # Agent's observation range
a_co = ((2 * a_range + 1) ** 2) * 4  # Index for agent coordinates in observation


# Initialize total rewards
total_reward = [float(0)] * n_agents
avg_reward = []
episode_reward = []

# Initialize environment and Q-values
env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range, 'linear')

obs_ = env.reset()

Q = np.zeros((n_agents, gridsize[1], gridsize[0], n_actions), dtype=float)

policy = np.zeros((n_agents, gridsize[1], gridsize[0]), dtype=int)
#--------------------------------------------------------------------------------------------------------



def plot(avg_reward, ep_reward):
    ep_rd = np.array(ep_reward)
    avg_rd = np.array(avg_reward)
    length = np.arange(1, n_episodes+1)
    
    plt.figure(figsize=(12, 6))
    for i in range (n_agents):
        color = randomcolor.RandomColor().generate()  
          
        plt.plot(length, ep_rd[:,i], color= color[0],  alpha=0.4 )

        plt.plot(length, avg_rd[:,i], color=color[0], label= f'Cumulative Reward for agent_{i}' )

    
    plt.title('Episode vs Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    
    plt.legend()
    
    plt.savefig(f"{n_agents}_{gridsize}_full_obs.jpg")
    
    plt.show()


# Main function to run the simulation
def main():
    for episode in range(n_episodes):
        obs_ = env.reset()
        state = np.array([obs_[i][a_co:a_co + 2] for i in range(n_agents)], dtype=int)
        
        # plate_found = [False] * n_agents 
        
        ep_reward = [float(0)] * n_agents  # Track rewards for this episode
        rd = [float(0)] * n_agents # Track reward for this episode
        
        for step in range(max_steps):
            
            
                
            temp_act = [4] * n_agents  # Initialize actions as NOOP
            
            for i in range(n_agents):
                temp_act[i] = random.choice([0,1,2,3,4])
            
            obs, rewards, done, _ = env.step(temp_act)
            
            for agent in range(n_agents):
                ep_reward[agent] += rewards[agent]
            
            # Update policy with selected action
                policy[agent, state[agent][0], state[agent][1]] = temp_act[agent]
                
                state[agent] = np.array(obs[agent][a_co:a_co + 2], dtype=int)
            
            if all(done):
                # print(f"Episode: {episode} finished")
                break
            
            # if episode > n_episodes-10:
            # env.render()
            # Uncomment to slow down rendering
            # time.sleep(0.05)
                
        print(f"Episode: {episode}")
        # Update total and average rewards
        for i in range(n_agents):
            total_reward[i] += ep_reward[i]
            rd[i] = total_reward[i]/(episode + 1)
        
            
        avg_reward.append(rd)

        episode_reward.append(ep_reward)
    
    # Plot the results
    plot(avg_reward, episode_reward)
        
if __name__ == "__main__":
    main()

avg_reward = np.array(avg_reward)
episode_reward = np.array(episode_reward)

np.save(f'{n_agents}_{gridsize}_q_roll_full_avg_reward.npy', avg_reward)
np.save(f'{n_agents}_{gridsize}_q_roll_full_episode_reward.npy', episode_reward)
np.save(f'{n_agents}_{gridsize}_q_roll_full_policy.npy', policy)
