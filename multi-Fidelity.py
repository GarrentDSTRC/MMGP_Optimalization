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
pathx2=r".\ROM\E3\saveX_gpytorch_multi_EI_MS.npy"
pathy2=r".\ROM\E3\savey_gpytorch_multi_EI_MS.npy"

path1H=r".\Database\saveX_gpytorch_multi_EI_MS_H.npy"
path2H=r".\Database\savey_gpytorch_multi_EI_MS_H.npy"
UPBound=np.array([1.0,0.6,40,180,9,9,35]).T
LOWBound=np.array([0.6,0.1,5,0,0,0,10]).T
# We make an nxn grid of training points spaced every 1/(n-1) on [0,1]x[0,1]
# n = 250
init_sample = 30
UpB=len(Frame.iloc[:, 0].to_numpy())-1
LowB=0
Infillpoints=20
training_iterations = 5#33#5
num_tasks=-2
num_input=len(UPBound)
Episode=1
LowSample=1800
testsample=140
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")
Offline=0
testnum=50
dict = [i for i in range(TestX.shape[0])]

if os.path.exists(path1):
    full_train_x=torch.FloatTensor(np.load(path1, allow_pickle=True))
    full_train_i=torch.FloatTensor(np.load(path2,allow_pickle=True))
    full_train_y=torch.FloatTensor(np.load(path3, allow_pickle=True))
    #full_train_y[:,1] = 10*full_train_y[:,1]
    if os.path.exists(path4):
        dict = np.load(path4, allow_pickle=True).astype(int).tolist()
else:
    train_x1 = torch.tensor(np.load(pathx2, allow_pickle=True)).to(torch.float32)
    train_y1 =torch.tensor( np.load(pathy2, allow_pickle=True)).to(torch.float32)

    # Hight fidelity
    sp = samplingplan(num_input)
    X = sp.optimallhc(init_sample)
    X= LOWBound+X*(UPBound-LOWBound)
    train_x2=np.zeros([init_sample, num_input])
    train_y2 = np.zeros([init_sample,abs(num_tasks)])
    size=len(Frame.iloc[:, 4].to_numpy())



    if Offline == 1:
        for index, value in enumerate(X):
            train_x2[index, :], train_y2[index, :] = findpoint_interpolate(value, Frame, num_tasks=num_tasks)
            if np.isnan(train_y2[index, 0]) or np.isnan(train_y2[index, 1]):
                train_x2[index, :], train_y2[index, :] = findpoint_interpolate(value, Frame, num_tasks=num_tasks,
                                                                               method="nearest")
    elif Offline == 0:
        ##online
            initialDataX, initialDataY = findpointOL(X,num_task=num_tasks)
    else :
        train_x2 = torch.tensor(np.load(path1H, allow_pickle=True)).to(torch.float32)
        train_y2 = torch.tensor(np.load(path2H, allow_pickle=True)).to(torch.float32)

    train_x2 = torch.tensor(train_x2).to(device).to(torch.float32)
    train_y2 = torch.tensor(train_y2).to(device).to(torch.float32)

    # Construct data

    train_i_task1 = torch.full((train_x1.shape[0], 1), dtype=torch.long, fill_value=0) #low
    train_i_task2 = torch.full((train_x2.shape[0], 1), dtype=torch.long, fill_value=1) #high

    # full_train_x = torch.cat([train_x1, train_x2]).to(device)
    # full_train_i = torch.cat([train_i_task1, train_i_task2]).to(device)
    # full_train_y = torch.cat([train_y1, train_y2]).to(device)
    # Construct data2

# Here we have two iterms that we're passing in as train_inputs
likelihood1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
#50: 0:2565
model1 = MultiFidelityGPModel((full_train_x[:-testnum,:], full_train_i[:-testnum,:]), full_train_y[:-testnum,0], likelihood1).to(device)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model2 = MultiFidelityGPModel((full_train_x[:-testnum,:], full_train_i[:-testnum,:]), full_train_y[:-testnum,1], likelihood2).to(device)
model = gpytorch.models.IndependentModelList(model1, model2).to(device)
likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)

print(model)

# #----------------------------------------- cuda
# full_train_x = full_train_x.cuda()
# full_train_i = full_train_i.cuda()
# full_train_y = full_train_y.cuda()

# model = model.cuda()
# likelihood = likelihood.cuda()
# #----------------------------------------- cuda
# "Loss" for GPs - the marginal log likelihood
from gpytorch.mlls import SumMarginalLogLikelihood
mll = SumMarginalLogLikelihood(likelihood, model)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters



for i in range(Episode):
    print(  "Episode%d-point %d : %d "%(i, torch.sum(full_train_i).item()-testnum,len(full_train_i)-torch.sum(full_train_i).item())   )
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

    #TEST测试
    # sp = samplingplan(num_input)
    # X = sp.optimallhc(8)
    # X= LOWBound+X*(UPBound-LOWBound)
    #X=torch.FloatTensor(np.load(path1H, allow_pickle=True))
    X=full_train_x[-testnum:,:]

    test_x=torch.tensor(X).to(device)
    test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_y2 = likelihood(*model((test_x.to(torch.float32), test_i_task2), (test_x.to(torch.float32), test_i_task2)))
        observed_pred_y21 = observed_pred_y2[0].mean
        observed_pred_y22 = observed_pred_y2[1].mean
    # test_y_actual1 = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
    print("测试点预测值(推力：效率)",observed_pred_y21,observed_pred_y22*0.1)

    #test_x, test_y_actual = findpointOL(X, num_task=num_tasks)
    #test_y_actual=torch.FloatTensor(np.load(path2H, allow_pickle=True))
    test_y_actual = full_train_y[-testnum:, :]
    print("测试点真实值(推力：效率)", test_y_actual)
    delta_y11 = torch.abs(observed_pred_y21 - test_y_actual[:,0]).detach().numpy()
    delta_y12 = torch.abs(observed_pred_y22 - test_y_actual[:,1]).detach().numpy()
    print("各测试点MAE测试误差百分比", delta_y11 * 100 / 40, "% 和", delta_y12 * 100 / 2 * 0.1, "%")

    print("MAE测试平均误差",np.mean(delta_y11),np.mean(delta_y12))
    print("MAE测试平均误差百分比", np.mean(delta_y11)*100/18,"% 和", np.mean(delta_y12)*100/17,"%")


