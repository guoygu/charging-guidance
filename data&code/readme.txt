This is the related code for the paper "Multiagent Proximal Policy Optimization for Dynamic Electric Vehicle Charging Guidance."

The MAPPO+GCN+A.py script includes both training and testing. If the line `return_list = train_mappo_agent(agent, num_episodes, replay_buffer)` is not commented out,The training will be conducted first, followed by testing. If this statement is commented out, only testing will be performed.

Additionally, during training and testing, it is necessary to first modify the routing file in train(railway).sumocfg to train(railway).rou.xml.