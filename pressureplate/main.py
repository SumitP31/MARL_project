from environment import PressurePlate
import gym
import time
import numpy as np
import random



n_episodes = 20
max_steps = 500
gridsize = (15,9)
n_agents = 4
n_actions = 5
a_range = 2
a_co = ((2*(a_range) + 1)**2)*4 # index for coodrinate of agent

# Initialize the environment
env = PressurePlate(gridsize[0], gridsize[1], n_agents, a_range,'linear')

obs_ = env.reset()

#  Initialize Q values
Q = np.zeros((n_agents, gridsize[1], gridsize[0], n_actions), dtype = float)

policy = np.zeros((n_agents, gridsize[1], gridsize[0]), dtype = int)

alpha = 1e-2 #laerning rate
gamma = 1 #penalty

epsilon = 1e-1 # espsilon for spsilon greedy policy
threshold  = 2e4 #converging threshold for episodic reward value


def policy_eval(state):
    actions = []
    
    for i in range(n_agents):
        x, y = state[i][0], state[i][1]
        actions.append(policy[i, x, y])
    return actions
    



def q_value(Q, state, agent):
       
    for a in range(n_actions):
        st = state.copy()

        next_st = state_transition(st, a) 
        reward = agent_reward(next_st, agent)
       
        Q[st[0], st[1], a] += alpha * (reward + gamma * np.max(Q[next_st[0], next_st[1]]) - Q[st[0], st[1], a])
    
    action = np.argmax(Q[state[0],state[1]])
        
    return action, Q    

def agent_reward(next_state, agent):
    if agent == 3:
        goal_loc = env.goal.x, env.goal.y
    else:
        goal_loc = env.plates[agent].x, env.plates[agent].y  
    
    curr_room = env._get_curr_room_reward(next_state[1])
    # print(f"\nCurrent room: {curr_room}")
    agent_loc = next_state[0], next_state[1]

    if agent == curr_room:
        reward = - np.linalg.norm((np.array(goal_loc) - np.array((agent_loc))), 1) / env.max_dist
        # print(reward)
    else:
        reward = -len(env.room_boundaries)+1 + curr_room   
    
    return reward
        
    

def state_transition(state, action):
    stat = state.copy()
    proposed_pos = [state[0],state[1]]
    # print(f"\n{proposed_pos}")
    if action == 0:
        proposed_pos[1] -= 1
        if not env._detect_collision(proposed_pos):
            stat[1] -= 1

    elif action == 1:
        proposed_pos[1] += 1
        if not env._detect_collision(proposed_pos):
            stat[1] += 1

    elif action == 2:
        proposed_pos[0] -= 1
        if not env._detect_collision(proposed_pos):
            stat[0] -= 1

    elif action == 3:
        proposed_pos[0] += 1
        if not env._detect_collision(proposed_pos):
            stat[0] += 1

    else:
        # NOOP
        pass
    
    return stat
        
def main():
    for episode in range(n_episodes):
        
        obs_ = env.reset()
        state = np.array([obs_[i][a_co:a_co+2] for i in range(n_agents)], dtype = int)
     
    
        
        for step in range(max_steps):
            
           
            
            
            for agent in range(n_agents):
                temp_act = [4,4,4,4]
                
                action, q = q_value(Q[agent], state[agent], agent) # Update Q values for each agent and output best action based on that
                
                temp_act[agent] = action
                
                obs, rewards, done, _ = env.step(temp_act)
                
                policy[agent,state[agent][0],state[agent][1]] = action
                
                
                
               
                
                state[agent] = np.array(obs[agent][a_co:a_co+2],dtype=int)
                            
                
                
                if all(done)==True:
                    print(f"Episode: {episode} finished----------------------------------------------------------")                    
                    break
                
                env.render()
                # time.sleep(0.1)
                
                
if __name__ == "__main__":
    main()
    
    
    
