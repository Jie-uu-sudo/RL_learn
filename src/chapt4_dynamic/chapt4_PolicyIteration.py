import copy
from src.chapt4_dynamic.chapt4_dynamic_CliffwalkingEnv import CliffWalkingEnv

class PolicyIteration:
    """策略迭代算法"""
    def __init__(self,env,theta,gamma):
        self.env=env
        self.v=[0]*self.env.ncol*self.env.nrow#状态价值
        self.pi=[[0.25,0.25,0.25,0.25] for i in range(self.env.ncol*self.env.nrow)]#初始策略
        self.theta=theta#评价收敛阈值
        self.gamma=gamma

    def policy_evaluation(self):#策略评估
        cnt=1#计数
        while 1:
            max_diff=0
            new_v=[0]*self.env.ncol*self.env.nrow
            for s in range(self.env.ncol*self.env.nrow):#遍历每个状态
                qsa_list=[]
                for a in range(4):#每个状态的动作
                    qsa=0
                    for res in self.env.P[s][a]:
                        p,next_state,r,done=res
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                    qsa_list.append(self.pi[s][a]*qsa)
                new_v[s]=sum(qsa_list)
                max_diff=max(max_diff,abs(new_v[s]-self.v[s]))
            self.v=new_v
            if max_diff<self.theta:
                break
            cnt+=1
            print("策略评估进行%d轮后完成"%cnt)

    def policy_improvement(self):#策略迭代
        for s in range(self.env.nrow*self.env.ncol):#遍历48个状态
            qsa_list=[]
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p,next_state,r,done=res
                    qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))#计算当前动作状态价值
                qsa_list.append(qsa)
            maxq=max(qsa_list)#
            cntq=qsa_list.count(maxq)#找出有几个动作是最大的
            #因为可能会出现多个动作最优，最优动作平分概率
            self.pi[s]=[1/cntq if q==maxq else 0 for q in qsa_list]
        print('策略提升完成')
        return self.pi
    
    def policy_iteration(self):#策略迭代
        while 1:
            self.policy_evaluation()
            old_pi=copy.deepcopy(self.pi)
            new_pi=self.policy_improvement()
            if old_pi == new_pi: break
            #前一策略等于当前更新的策略就表示收敛结束

def print_agent(agent,action_meaning,disaster=[],end=[]):
    print("状态价值:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s'%('%.3f'%agent.v[i*agent.env.ncol+j]),end=' ')
        print()
    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i*agent.env.ncol+j) in disaster:#掉下悬崖
                print("****",end=" ")
            elif(i*agent.env.ncol+j) in end:#到达终点
                print("EEEE",end=" ")
            else:
                a=agent.pi[i*agent.env.ncol+j]
                pi_str=""
                for k in range(len(action_meaning)):
                    pi_str+=action_meaning[k] if a[k] >0 else "o"
                print(pi_str,end=" ")
            print()

env=CliffWalkingEnv()
action_meaning=["^","|","<",">"]
theta=0.001
gamma=0.9
agent=PolicyIteration(env,theta,gamma)
agent.policy_iteration()
print_agent(agent,action_meaning,list(range(37,47)),[47])




