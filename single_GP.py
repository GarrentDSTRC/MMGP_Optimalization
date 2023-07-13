import math
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import gpytorch
from matplotlib import pyplot as plt
from pyKriging.samplingplan import samplingplan
import pandas as pd
from time import time
from scipy.interpolate import griddata
import os
import numpy as np
import random
from GPy import *
import numpy as np

# path1=r".\saveX_gpytorch_multi.npy"
# path2=r".\savey_gpytorch_multi.npy"
# path3=r".\saveTestX_gpytorch_multi.npy"

#path1=r".\RAM\saveX_gpytorch_multi_EI_MS.npy"
#path2=r".\RAM\savey_gpytorch_multi_EI_MS.npy"
path3=r".\Database\saveTestXdict_gpytorch_multi_EI_MS.npy"

path1=r".\Database\saveX_gpytorch_multifidelity_multitask.npy"
path2=r".\Database\saveY_gpytorch_multifidelity_multitask.npy"
path1H=r".\Database\saveX_gpytorch_multi_EI_MS_H.npy"
path2H=r".\Database\savey_gpytorch_multi_EI_MS_H.npy"
path5='.\Database\singlefidelity_high_database.pth'
#path5='.\Database\singlefidelity_low_database.pth'
UPBound=np.array([1.0,0.6,40,180,9,9,35]).T
LOWBound=np.array([0.6,0.1,5,0,0,0,10]).T
init_sample=8*4
training_iter=5#110
Infillpoints=8*2
Episode=1#47
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
num_tasks=2
Offline=0
testnum=50
dict = [i for i in range(TestX.shape[0])]

if __name__=="__main__":
    if os.path.exists(path1):
        initialDataX=np.load(path1,allow_pickle=True)
        initialDataY=np.load(path2,allow_pickle=True)
        #initialDataY[:, 1] = 10 * initialDataY[:, 1]
        dict=np.load(path3,allow_pickle=True).astype(int).tolist()
        for i in range(initialDataX.shape[0]):
            for j in range(initialDataX.shape[1]):
                initialDataX[i,j]=round(initialDataX[i,j],1)
    else:
        sp = samplingplan(7)
        X = sp.optimallhc(init_sample)
        X= LOWBound+X*(UPBound-LOWBound)
        for i in range(init_sample):
            for j in range(4,7):
                X[i,j]=int( X[i,j])

        initialDataX=np.zeros([init_sample,7])
        initialDataY = np.zeros([init_sample,np.abs(num_tasks)])
        if Offline == 1:
            for index, value in enumerate(X):
                initialDataX[index, :], initialDataY[index:index + 1, :] = findpoint_interpolate(value, Frame, 2)
                if np.isnan(initialDataY[index, 0]) or np.isnan(initialDataY[index, 1]):
                    initialDataX[index, :], initialDataY[index, :] = findpoint_interpolate(value, Frame, 2, "nearest")
        else:
            ##online
            initialDataX, initialDataY = findpointOL(X,num_task=2)


    train_x=torch.tensor(initialDataX).to(device).to(torch.float32)
    train_y=torch.tensor(initialDataY).to(device).to(torch.float32)
    # independent Multitask
    likelihood1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model1 = SpectralMixtureGPModel(train_x[0:2216,:],train_y[0:2216,0], likelihood1).to(device)
    likelihood2 = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model2 = SpectralMixtureGPModel(train_x[0:2216,:],train_y[0:2216,1], likelihood2).to(device)
    model = gpytorch.models.IndependentModelList(model1, model2).to(device)
    likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)
    from gpytorch.mlls import SumMarginalLogLikelihood
    mll = SumMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    cofactor = [0.5, 0.5]
    for j in range(Episode):
        print("Episode",j,"point",len(train_y))
        model.train()
        likelihood.train()
        if os.path.exists(path5):
            state_dict = torch.load(path5)
            model.load_state_dict(state_dict)
        else:
            for i in range(training_iter):
                optimizer.zero_grad()
                output = model(*model.train_inputs)
                loss = -mll(output, model.train_targets)
                loss.backward(retain_graph=True)
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
                optimizer.step()
            torch.save(model.state_dict(), path5)

        #################test###################
        X = torch.FloatTensor(np.load(path1H, allow_pickle=True))

        test_x = torch.tensor(X).to(device)
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_y2 = likelihood(
                *model(test_x.to(torch.float32), test_x.to(torch.float32)))
            observed_pred_y21 = observed_pred_y2[0].mean
            observed_pred_y22 = observed_pred_y2[1].mean
        # test_y_actual1 = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
        print("测试点预测值(推力：效率)", observed_pred_y21, observed_pred_y22 * 0.1)

        test_y_actual = torch.FloatTensor(np.load(path2H, allow_pickle=True))
        print("测试点真实值(推力：效率)", test_y_actual)
        delta_y11 = torch.abs(observed_pred_y21 - test_y_actual[:, 0]).detach().numpy()
        delta_y12 = torch.abs(observed_pred_y22 - test_y_actual[:, 1]).detach().numpy()

        print("MAE测试平均误差", np.mean(delta_y11), np.mean(delta_y12))
        print("MAE测试平均误差百分比", np.mean(delta_y11) * 100 / 18, "% 和", np.mean(delta_y12) * 100 / 17, "%")
        ###########################################################test###################
        # X,Y=infillGA(model, likelihood, Infillpoints, dict, num_tasks,"EI", device=device, cofactor=cofactor, y_max=[torch.max(train_y[:,0]).item() ,torch.max(train_y[:,1]).item()], offline=Offline,train_x=train_x)
        # cofactor=UpdateCofactor(model,likelihood,X.to(torch.float32),Y.to(torch.float32),cofactor,torch.max(train_y,dim=0).values-torch.min(train_y,dim=0).values)
        # #cofactor=[0.5,0.5]
        # print("addpoint",X)
        # train_x=torch.cat((train_x,X),dim=0).to(torch.float32)
        # train_y=torch.cat((train_y,Y),dim=0).to(torch.float32)
        # # #correlated Multitask
        # # model = MultitaskGPModel(train_x, train_y, likelihood).to(device)
        # # independent Multitask
        # model1 = SpectralMixtureGPModel(train_x, train_y[:, 0], likelihood1).to(device)
        # model2 = SpectralMixtureGPModel(train_x, train_y[:, 1], likelihood2).to(device)
        # model = gpytorch.models.IndependentModelList(model1, model2).to(device)
        #
        # model.train()
        # likelihood.train()
        #
        # if j % 1 == 0:
        #     np.save(path1, np.array(train_x.cpu()))
        #     np.savetxt(r'.\RAM\train_x.csv', np.array(train_x.cpu()), delimiter=',')
        #     np.savetxt(r'.\RAM\train_y.csv', np.array(train_y.cpu()), delimiter=',')
        #     np.save(path2, np.array(train_y.cpu()))
        #     np.save(path3, np.array(dict))
        #
        #     np.save(r".\ROM\E3\saveX_gpytorch_multi_EI_MS %d.npy" % (len(train_y)), np.array(train_x.cpu()))
        #     np.save(r".\ROM\E3\savey_gpytorch_multi_EI_MS %d.npy" % (len(train_y)), np.array(train_y.cpu()))
        #     np.save(r".\ROM\E3\saveTestXdict_gpytorch_multi_EI_MS %d.npy" % (len(train_y)), np.array(dict))





