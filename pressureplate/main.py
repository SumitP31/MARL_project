from environment import PressurePlate

env = PressurePlate(height=15, width=9, n_agents=4, sensor_range=4, layout='linear')
obs = env.reset()

action = [3,3,3,3] # implement your own algorithm for choosing actions

''' returns each agent's observation matrix of form (no. of agents x (agents_pos + wall_pos + door_pos + plate_pos + agent_coord)) 
                        rewards for each agent i.e.; list of four element [r1,r2,r3,r4]
                        termination status of loop i.e.; if goal is reached'''
obs_, reward_, terminated, info = env.step(action)

print(f"Observations : \n{obs_}\n reward : \n{reward_}\n ")