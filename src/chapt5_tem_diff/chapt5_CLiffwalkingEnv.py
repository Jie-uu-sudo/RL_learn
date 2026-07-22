import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffwalkingEnv:
    def __init__(self,ncol,nrow):
        self.ncol=ncol
        self.nrow=nrow
        self.x=0#初始横坐标
        self.y=self.nrow-1#初始纵坐标

    def step(self,action):#外部调用该函数改变当前位置
        change=[[0,-1],[0,1],[-1,0],[1,0]]
        self.x=min(self.ncol-1,max(0,self.x+change[action][0]))#防止超出最大最小界限
        self.y=min(self.nrow-1,max(0,self.y+change[action][1]))
        next_state=self.x+self.y*self.ncol#下一状态
        reward=-1#每走一步设定奖励为-1
        done=False#当前未完成
        #下一位置在悬崖或者终点
        if self.y==self.nrow-1 and self.x>0:
            done=True
            if self.x != self.ncol-1:
                #下一步在悬崖
                reward=-100
        return next_state,reward,done
    def reset(self):
        self.x=0
        self.y=self.nrow-1
        return self.x+self.y*self.ncol
    

        