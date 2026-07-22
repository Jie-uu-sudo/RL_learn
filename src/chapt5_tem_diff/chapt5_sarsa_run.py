from chapt5_tem_diff.chapt5_CLiffwalkingEnv import CliffwalkingEnv
from chapt5_tem_diff.chapt5_sarsa import Sarsa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ncol=12
nrow=4
env=CliffwalkingEnv(ncol,nrow)
np.random.seed(0)
epsilon=0.1
alpha=0.1
gamma=0.9
agent=Sarsa(ncol,nrow,epsilon,alpha,gamma)
num_episodes=500#智能体在环境中运行的序列的数量
 
return_list=[]
for i in range(10):
    with tqdm(total=int(num_episodes/10),desc="Iteration%d"%i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return=0
            state=env.reset()
            action=agent.take_action(state)
            done=False
            while not done:
                next_state,reward,done=env.step(action)
                next_action=agent.take_action(next_state)
                episode_return+=reward
                agent.update(state,action,reward,next_state,next_action)
                state=next_state
                action=next_action
            return_list.append(episode_return)
            if(i_episode+1)%10==0:
                pbar.set_postfix({'episode':'%d'%(num_episodes/10*i+i_episode+1),'return':'%.3f'%np.mean(return_list[-10:])})
            pbar.update(1)

episode_list=list(range(len(return_list)))
plt.plot(episode_list,return_list)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Sarsa on {}".format("Cliff Walking"))
plt.show()

