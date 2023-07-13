import math
import numpy as np
import torch
import gpytorch
from pyKriging.samplingplan import samplingplan
import time
import os
from GPy import *

from gpytorch import kernels, means, models, mlls, settings, likelihoods, constraints, priors
from gpytorch import distributions as distr
path5='.\Database\multifidelity_database.pth'

path1=r".\Database\saveX_gpytorch_multifidelity_multitask.npy"
path2=r".\Database\saveI_gpytorch_multifidelity_multitask.npy"
path3=r".\Database\saveY_gpytorch_multifidelity_multitask.npy"
path4=r".\Database\saveTestXdict_gpytorch_multifidelity_multitask.npy"

UPBound=np.array([1.0,0.6,40,180,9,9,35]).T
LOWBound=np.array([0.6,0.1,5,0,0,0,10]).T
# We make an nxn grid of training points spaced every 1/(n-1) on [0,1]x[0,1]
# n = 250
init_sample = 30
UpB=len(Frame.iloc[:, 0].to_numpy())-1
LowB=0
Infillpoints=20
training_iterations = 33#33#5
num_tasks=-2
num_input=len(UPBound)
Episode=1

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")
Offline=0

dict = [i for i in range(TestX.shape[0])]

if os.path.exists(path1):
    full_train_x=torch.FloatTensor(np.load(path1, allow_pickle=True))
    full_train_i=torch.FloatTensor(np.load(path2,allow_pickle=True))
    full_train_y=torch.FloatTensor(np.load(path3, allow_pickle=True))
    full_train_y[:,1] = 10*full_train_y[:,1]
    if os.path.exists(path4):
        dict = np.load(path4, allow_pickle=True).astype(int).tolist()

# Here we have two iterms that we're passing in as train_inputs
likelihood1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
#50: 0:2565
model1 = MultiFidelityGPModel((full_train_x[50:,:], full_train_i[50:,:]), full_train_y[50:,0], likelihood1).to(device)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model2 = MultiFidelityGPModel((full_train_x[50:,:], full_train_i[50:,:]), full_train_y[50:,1], likelihood2).to(device)
model = gpytorch.models.IndependentModelList(model1, model2).to(device)
likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)

print(model)


# "Loss" for GPs - the marginal log likelihood
from gpytorch.mlls import SumMarginalLogLikelihood
mll = SumMarginalLogLikelihood(likelihood, model)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters



for i in range(Episode):
    print(  "Episode%d-point %d : %d "%(i, torch.sum(full_train_i).item(),len(full_train_i)-torch.sum(full_train_i).item())   )
    cofactor = [0.5, 0.5]
    if os.path.exists(path5):
        state_dict = torch.load(path5)
        model.load_state_dict(state_dict)
    else:
        for j in range(training_iterations):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward(retain_graph=True)
            print('Iter %d/%d - Loss: %.3f' % (j + 1,training_iterations, loss.item()))
            optimizer.step()
        torch.save(model.state_dict(), path5)


    #a*i 排除=0
    index=int(full_train_i.shape[0]-torch.sum(full_train_i))
    y_max=[torch.max(full_train_y[index:,0]).item() ,torch.max(full_train_y[index:,1]).item()]

    #TEST测试 UPBound=[1.0,0.6,40,180,9,9,35]
       #St, y,  θ,  ψ,NACAm   p   t
    while(1):
        X=[[0.6, 0.1,5,  0,  0,    0,  10]]
        # 循环遍历X里的每个元素
        for i in range(len(X[0])):
            # 用input()函数获取用户的输入
            new_value = input(f"请输入第{i + 1}个值({LOWBound[i]}-{UPBound[i]}): ")
            # 用int()函数将输入转换为整数
            new_value = float(new_value)
            # 用新的值替换X里的元素
            X[0][i] = new_value
        # 打印修改后的X
        print(X)
        #X=UPBound

        test_x=torch.tensor(X).to(device).to(torch.float32)
        test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_y2 = likelihood(*model((test_x.to(torch.float32), test_i_task2), (test_x.to(torch.float32), test_i_task2)))
            observed_pred_y21 = observed_pred_y2[0].mean
            observed_pred_y22 = observed_pred_y2[1].mean
        # test_y_actual1 = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
        print("测试点预测值(推力：效率)",observed_pred_y21,observed_pred_y22*0.1)



