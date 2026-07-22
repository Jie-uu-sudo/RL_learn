import numpy as np
import matplotlib.pyplot as plt

class Bernoullibandit:
    def __init__(self,k):
        self.probs=np.random.uniform(size=k)#给每一个杆一个获奖的概率
        self.best_idx=np.argmax(self.probs)#找出概率最大的索引
        self.best_probs=self.probs[self.best_idx]#最大的概率的值
        self.K=k#老虎机的臂数


    def step(self,k):
        if np.random.rand()<self.probs[k]:
            return 1
        else:
            return 0
        
#np.random.seed(1)
K=10
bandit_10_arm=Bernoullibandit(K)
print("随机生成了一个%d臂伯努利老虎机"%K)
print("获奖概率最大为%d号拉杆，获奖概率为%.4f"%(bandit_10_arm.best_idx+1,bandit_10_arm.best_probs))


class Solver:
    """多臂老虎机的基本算法框架"""
    def __init__(self,bandit):
        self.bandit=bandit
        self.counts=np.zeros(self.bandit.K)#记录每根杆被拉了几次
        self.regret=0.#累计遗憾
        self.actions=[]#记录每次拉了哪根杆
        self.regrets=[]#每一步的遗憾值

    def update_regret(self,k):#更新以及累计遗憾
        self.regret+=self.bandit.best_probs-self.bandit.probs[k]
        self.regrets.append(self.regret)
    
    def run_one_step(self):
        raise NotImplementedError
    
    def run(self,num_steps):
        for i in range(num_steps):
            k=self.run_one_step()
            self.counts[k]+=1
            self.actions.append(k)
            self.update_regret(k)
            if i%100==0:
                print("当前选择的是第%d根杆"%(k+1))
            if i==num_steps-1:
                print("当前轮次结束，最终选择%d杆"%(k+1))

class UCB(Solver):
    def __init__(self, bandit,coef,init_prob=1.0):
        super(UCB,self).__init__(bandit)
        self.total_count=0
        self.estimates=np.array([init_prob]*self.bandit.K)
        self.coef=coef
    
    def run_one_step(self):
        self.total_count+=1
        ucb=self.estimates+self.coef*np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))
        k=np.argmax(ucb)
        r=self.bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])
        return k


def plot_results(solvers,solver_names):
    for idx,solver in enumerate(solvers):
        time_list=range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label=solver_names[idx])
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title("%d-armed bandit"%solvers[0].bandit.K)
    plt.legend()
    plt.show()

UCB_solver=UCB(bandit_10_arm,1)
UCB_solver.run(5000)
print("total_regret:",UCB_solver.regret)
plot_results([UCB_solver],["UCB"])
