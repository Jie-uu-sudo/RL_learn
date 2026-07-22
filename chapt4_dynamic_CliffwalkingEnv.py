class CliffWalkingEnv:
    """悬崖漫步环境"""
    def __init__(self,ncol=12,nrow=4):
        self.ncol=ncol#列,x
        self.nrow=nrow#行,y
        self.P=self.createP()#转移矩阵

    def createP(self):
        P=[[[]for j in range(4)] for i in range(self.nrow*self.ncol)]
        #P是一个48行4列的转移矩阵,每一个元素都包含(p,next_state,reward,done)，列代表的是4种动作
        change=[[0,-1],[0,1],[-1,0],[1,0]]#4种动作
        for i in range(self.nrow):
            for j in range(self.ncol):#遍历每一个状态
                for a in range(4):
                    if i==self.nrow-1 and j>0:
                        P[i*self.ncol+j][a]=[(1,i*self.ncol+j,0,True)]
                        continue
                    next_x=min(self.ncol-1,max(0,j+change[a][0]))#防止超出最大最小界限
                    next_y=min(self.nrow-1,max(0,i+change[a][1]))
                    next_state=next_x+next_y*self.ncol#下一状态
                    reward=-1#每走一步设定奖励为-1
                    done=False#当前未完成
                    #下一位置在悬崖或者终点
                    if next_y==self.nrow-1 and next_x>0:
                        done=True
                        if next_x != self.ncol-1:
                            #下一步在悬崖
                            reward=-100
                    P[i*self.ncol+j][a]=[(1,next_state,reward,done)]
        return P

