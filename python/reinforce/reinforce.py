import torch
import numpy as np

from torch.autograd import Variable


def agent_ttt(act_agent_prev, act_user_prev):
    return act_user_prev
    
def agent_evil(act_agent_prev, act_user_prev):
    return 0
    
def agent_nice(act_agent_prev, act_user_prev):
    return 1
        
def reward(act_agent, act_user):
    '''Return the reward for the user
       act_agent -- the action of the agent (0 is defect, 1 is cooperate)
       act_user --  the action of the user (0 is defect, 1 is cooperate)
    '''
    # 0: defect
    # 1: cooperate
    # matrix:
    #   
    
    #   agent             user
    #
    #               defect    cooperate
    #  defect       -2, -2         0, -3
    #  cooperate    -3,  0        -1, -1            
    #
    reward_matrix = {(0, 0): -2.0,
                     (1, 1): -1.0,
                     (0, 1): -3.0,
                     (1, 0):  0.0}
    
    return reward_matrix[(act_agent, act_user)]
    
    
    
    

map_states =  {(0, 0): 0,
               (1, 1): 1,
               (0, 1): 2,
               (1, 0): 3}

q = Variable(torch.randn(4), requires_grad=True)   # P(cooperate) for
                                                   # (0, 0), (1, 1), (0, 1), (1, 0)
                                                   # in the previous game



N = 1000
n_iter = 100


learning_rate = 0.0001

for i in range(N):
    act_agent = np.random.randint(2)
    act_user = np.random.randint(2)
    G = []
    logPa = [] 
    a = []
    
    agent_fn = [agent_ttt, agent_evil, agent_nice][np.random.randint(3)]
    
    
    for iter in range(n_iter):
        
        prob_cur = torch.sigmoid(q[map_states[(act_agent, act_user)]])
        
        act_agent, act_user = agent_fn(act_agent, act_user), np.random.binomial(1, prob_cur.data.numpy()[0])
        
        g_cur = reward(act_agent, act_user)
        
        a.append(act_user)
        G.append(g_cur)
        if act_user == 0:
            logPa.append(torch.log(1-prob_cur))
        else:
            logPa.append(torch.log(prob_cur))
    
    returns = np.cumsum(G[::-1])[::-1]
    
    for iter in range(n_iter):
        weighted_pa = logPa[iter] * returns[iter]
        weighted_pa.backward()
        
    
    #print(q.grad.data.numpy())
    
    q.data = q.data + learning_rate * q.grad.data
    q.grad.data.zero_()
    
    print(q.data.numpy())
    
    print("Reward", returns[0])
        
    
    
        
        
        
    
    






