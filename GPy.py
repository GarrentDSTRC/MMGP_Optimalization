import torch
torch.set_default_tensor_type(torch.FloatTensor)
import gpytorch
import pandas as pd
from time import time
from scipy.interpolate import griddata
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import norm
import random
from pprint import pprint
from gpytorch.priors import NormalPrior
from deap import algorithms, base, creator, tools
from functools import partial
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
from matplotlib import pyplot as plt
Frame = pd.read_excel('.\ROM\BF_search.xlsx', sheet_name="HL")
Frame2 = pd.read_excel('.\ROM\BF_search.xlsx', sheet_name="HL")
#CT2ETA2_CUT; ETA_CUT
#TestX = torch.FloatTensor(Frame.iloc[:, 0:4].to_numpy()).to(device)
St = torch.linspace(0.6,1.0,5)
ad = torch.linspace(0.1,0.6,6)
phi = torch.linspace(5,40,8)
theta = torch.linspace(0,180,7)
N = torch.linspace(0,9,4)
A = torch.linspace(0,9,4)
CA = torch.linspace(10,35,6)
a,b,c,d,e,f,g=torch.meshgrid(St,ad,phi,theta,N,A,CA)
TestX=torch.as_tensor(list(zip(a.flatten(),b.flatten(),c.flatten(),d.flatten(),e.flatten(),f.flatten(),g.flatten())) )
OLSCALE=10
UPB=[1.0,0.6,40,180,9,9,35]
LOWB=[0.6,0.1,5,0,0,0,10]
STP=[5,6,8,7,4,4,6]
import time
inittime=time.time()
from gpytorch.kernels import Kernel

from linear_operator.operators import (
    DiagLinearOperator,
    InterpolatedLinearOperator,
    PsdSumLinearOperator,
    RootLinearOperator,
)
"-----------------------FIND POINT-----------------------------------------------"
def findpoint(point,Frame):
    min=99
    minj=0
    for j in range(len(Frame.iloc[:,0])):
        abs=np.sum(np.abs(Frame.iloc[j,0:4].to_numpy()-point))
        if min>abs:
            min=abs
            minj=j
    return Frame.iloc[minj,0:4].to_numpy(),Frame.iloc[minj,4]
def findpointOL(X,num_task=1):
    num_p=X.shape[0]
    all_Y=[]
    num_task=np.abs(num_task)
    for i in range(int(num_p/8)):
        for j in range(8):
            np.savetxt(r'.\MMGP_OL%d\dataX.txt' % (j % 8), X[i * 8 + j:i * 8 + j + 1, :], delimiter=',', fmt='%.02f,%.02f,%.02f,%.02f,%d%d%d')
            np.savetxt(r'.\MMGP_OL%d\flag.txt'%(j%8), np.array([0]), delimiter=',', fmt='%d')

        for j in range(8):
            flag=np.loadtxt(r'.\MMGP_OL%d\flag.txt'%(j%8), delimiter=",", dtype="int")

            while flag==0:
                try:
                    flag=np.loadtxt(r'.\MMGP_OL%d\flag.txt'%(j%8), delimiter=",", dtype="int")
                finally:
                    time.sleep(25)
                    print("程序运行时间",(time.time()-inittime)/3600)
            all_Y.append(np.loadtxt(r'.\MMGP_OL%d\dataY.txt'%(j%8), delimiter=",", dtype="float"))
    all_Y=np.asarray(all_Y)
    all_Y[:,1]=all_Y[:,1]*OLSCALE

    if num_task==1:
        return torch.tensor(X).to(device), torch.tensor(all_Y[:,0]).to(device)
    else:
        return torch.tensor(X).to(device), torch.tensor(all_Y).to(device)
def findpoint_interpolate(point,Frame,num_tasks=1,method="linear"):
    X=[]
    num_tasks=np.abs(num_tasks)
    for i in range(4):
        X.append(Frame.iloc[:,i].to_numpy())
    if num_tasks==1:
        Y=Frame.iloc[:,4].to_numpy()
        return point,griddata(tuple(X), Y, point,method=method)
    else:
        Y = Frame.iloc[:, 4:4+num_tasks].to_numpy()
        value=[]
        for i in range(num_tasks):
            value.append(griddata(tuple(X), Y[:,i], point, method=method))
        return point,np.array(value).T

def infill(model, likelihood, n_points, dict, num_tasks=1, method="error", cofactor=[0.5,0.5], offline=1, device = torch.device("cpu"),y_max=999):
    "num_task<0    ->use the multi-fidelity kernel"
    #num_task=1 Single GP; num_task=0 Raw;num_task=2 Multitask;num_task=-2  and multifidelity ;num_task=-1 multifidelity
    randomsample=3000
    Result_idx = []
    model.eval()
    likelihood.eval()
    print("num_dict",len(dict))
    selectDict= random.sample(dict, randomsample)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        if num_tasks<0:
            B=TestX[selectDict, :].to(device)
            C=torch.ones( len(TestX[selectDict, :]))
            if num_tasks==-2:
                A=likelihood(*model((B,C),(B,C) ))
                VarS = A[0].variance
                MeanS = A[0].mean
                VarS2=A[1].variance
                MeanS2=A[1].mean
            else:
                A=likelihood(model(B, C))
                VarS = A.variance
                MeanS = A.mean
        else:
            if num_tasks != 1:
                B = TestX[selectDict, :].to(device)
                A = likelihood(*model(B, B))
                VarS = A[0].variance
                MeanS = A[0].mean
                VarS2=A[1].variance
                MeanS2=A[1].mean
            else:
                A = likelihood(model(TestX[selectDict, :]))
                VarS = A.variance
                MeanS = A.mean
        if method == "EI":
            VarS=VarS+1e-05 #prevent var=0
            EI_one = (MeanS-y_max[0]) * torch.FloatTensor(norm.cdf(((MeanS-y_max[0])/VarS).cpu().detach())).to(device)
            EI_two = VarS* torch.FloatTensor(norm.pdf( (MeanS-y_max[0]/VarS).cpu().detach() )).to(device)
            EI = EI_one*cofactor[0] + EI_two*(1-cofactor[0])

            if np.abs(num_tasks)==2:
                VarS2 = VarS2 + 1e-05
                EI_one1 = (MeanS2-y_max[1]) * torch.FloatTensor(norm.cdf(((MeanS2-y_max[1]) / VarS2).cpu().detach())).to(device)
                EI_two1 = VarS2 * torch.FloatTensor(norm.pdf(((MeanS2-y_max[1]) / VarS2).cpu().detach())).to(device)
                EI1 = EI_one1*cofactor[0] + EI_two1*(1-cofactor[0])
                EI=(EI/torch.max(EI))*cofactor[1]+(EI1/torch.max(EI1))*(1-cofactor[1])


            VarS=EI
        if method == "PI":
            VarS=VarS+1e-05 #prevent var=0
            PI = torch.FloatTensor(norm.cdf(((MeanS-y_max[0])/VarS).cpu().detach())).to(device)

            if np.abs(num_tasks)==2:
                VarS2 = VarS2 + 1e-05
                PI1 =  torch.FloatTensor(norm.cdf(((MeanS2-y_max[1]) / VarS2).cpu().detach())).to(device)
                PI=(PI/torch.max(PI))*cofactor[1]+(PI1/torch.max(PI1))*(1-cofactor[1])
            VarS=PI
        if method == "UCB":
            k=1.68
            UCB=k*VarS+MeanS
            if np.abs(num_tasks) == 2:
                UCB1 = k * VarS2 + MeanS2
                UCB= (UCB / torch.max(UCB)) * cofactor[1] + (UCB1 / torch.max(UCB1)) * (1 - cofactor[1])
            VarS=UCB

        for i in range(n_points):
            Result_idx.append(selectDict[torch.argmax(VarS).item()])
            VarS[torch.argmax(VarS).item()] = -999
            print("remove", Result_idx[i])
        for i in range(n_points):
            dict.remove(Result_idx[i])
        if offline==1:
            return torch.tensor(Frame.iloc[Result_idx, 0:4].to_numpy()).to(device), torch.tensor(
            Frame.iloc[Result_idx, 4:4 + np.abs(num_tasks)].to_numpy()).to(device)
        else:
            X =TestX[Result_idx, :]
            X,Y=findpointOL(X,num_task=num_tasks)
            return X,Y

def infillGA(model, likelihood, n_points, dict, num_tasks=1, method="error", cofactor=[0.5,0.5], offline=1, device = torch.device("cpu"),y_max=999,train_x=[]):
    "num_task<0    ->use the multi-fidelity kernel"
    #num_task=1 Single GP; num_task=0 Raw;num_task=2 Multitask;num_task=-2  and multifidelity ;num_task=-1 multifidelity
    NDIM=7
    train_x=[ train_x[i,:].tolist() for i in range(train_x.shape[0])]
    for i,value in enumerate(train_x):
        train_x[i]=[round(j,2) for j in value]

    #trainx=[x for ]
    cxProb = 0.7
    mutateProb = 0.2
    popsize=500
    model.eval()
    likelihood.eval()
    print("num_dict",len(dict))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        creator.create("FitnessMax", base.Fitness, weights=(1, 1))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, 7)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=7)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluateEI, model=model, likelihood=likelihood,y_max=y_max,cofactor=cofactor)
        toolbox.decorate('evaluate', tools.DeltaPenalty(feasibleMT,-1e3))  # death penalty
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        # 运行遗传算法
        pop = toolbox.population(n=500)
        hof = tools.HallOfFame(32)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, stats=stats, halloffame=hof,
                                           verbose=True)
        #algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=100, cxpb=0.8, mutpb=1.0/NDIM, ngen=100)
        # 计算Pareto前沿集合
        for i in range(1,10):
            fronts = tools.emo.sortLogNondominated(pop, popsize,first_front_only=False)
            #pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof,
                                               #verbose=True)
            for front in fronts:
                tools.emo.assignCrowdingDist(front)
            pop = []
            for front in fronts:
                pop += front

            # 复制
            pop = toolbox.clone(pop)
            # 基于拥挤度的选择函数用来实现精英保存策略
            pop = tools.selNSGA2(pop, k=popsize, nd='standard')
            # 创建子代
            offspring = toolbox.select(pop, popsize)
            offspring = toolbox.clone(offspring)
            offspring = algorithms.varAnd(offspring, toolbox, cxProb, mutateProb)

            # 记录数据-将stats的注册功能应用于pop，并作为字典返回
            record = stats.compile(pop)
            logbook.record(gen=i, **record)

            # 合并父代与子代
            pop = pop + offspring
            # 评价族群-更新新族群的适应度
            fitnesses = map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
        print(logbook.stream)
            #pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, stats=stats, halloffame=hof,
                                               #verbose=True)
        pareto_front_ALL=tools.emo.sortLogNondominated(pop, popsize, first_front_only=False)
        # 计算IGD Spread
        #reference = pareto_front(n_points=100)
        #igd = tools.emo.igd(pareto_front, reference)
         # 四舍五入并选择16个不重复的candidate
        #gridPop
        for ind in front:
            plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'r.', ms=2)
        plt.xlabel('f_1')
        plt.ylabel('f_2')
        plt.tight_layout()

        plt.savefig("taskRelaiton.png", dpi=300)

        candidates = []
        for pareto_front in pareto_front_ALL:
            for ind in pareto_front:
                candidate = [round(x)*(UPB[i]-LOWB[i])/(STP[i]-1)+LOWB[i] for i,x in enumerate(ind)]
                if candidate not in candidates and candidate not in train_x:
                    candidates.append(candidate)
                if len(candidates) == n_points:
                    break
                if len(candidates) >= n_points:
                    candidates=candidates[0:n_points]
                    break
        if len(candidates) < n_points:
            candidates = candidates[0:len(candidates)-len(candidates)%8]

        X = torch.tensor(candidates).to(device).to(torch.float32)
        X,Y=findpointOL(X,num_task=num_tasks)
        return X,Y

def UpdateCofactor(model,likelihood,X,Y,cofactor,maxmin,MFkernel=0):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if MFkernel==0:
            A = likelihood(*model(X, X))
        else:
            I=torch.ones(X.shape[0]).to(torch.float32)
            A =likelihood(*model((X, I), (X, I)))
        M = torch.mean(torch.abs(Y - torch.cat([A[0].mean.unsqueeze(1), A[1].mean.unsqueeze(1)], dim=1)), dim=0)
        cofactor[1] = M[0] / maxmin[0] / (M[0] / maxmin[0] + M[1] / maxmin[1])

        f = open("./cofactor.txt", "a", encoding="utf - 8")
        f.writelines((str(cofactor[1].item()) + ",", str(M[0].item()) + ",", str(M[1].item()) + "\n"))
        f.close()

        if MFkernel == 0:
            return cofactor
        else: return cofactor,M
"-----------------------DIY KERNEL-----------------------------------------------"
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,mode="M"):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        if mode=="M":
        #self.covar_module =gpytorch.kernels.GridInterpolationKernel(
            #gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),grid_size=grid_size, num_dims=4)
            #self.covar_module = gpytorch.kernels.MaternKernel()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

        elif mode=="MR":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif mode=="R":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=train_x.shape[1])
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
class MultitaskGPModel(gpytorch.models.ExactGP):
    #model with output-covariance
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )

        # grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=2)
        #     , num_tasks=2, rank=1
        # )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
             gpytorch.kernels.RBFKernel()
             , num_tasks=2, rank=1
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
class TwoFidelityIndexKernel(Kernel):
    """
    Separate kernel for each task based on the Hadamard Product between the task
    kernel and the data kernel. based on :
    https://github.com/cornellius-gp/gpytorch/blob/master/examples/03_Multitask_GP_Regression/Hadamard_Multitask_GP_Regression.ipynb

    The index identifier must start from 0, i.e. all task zero have index identifier 0 and so on.

    If noParams is set to `True` then the covar_factor doesn't include any parameters.
    This is needed to construct the 2nd matrix in the sum, as in (https://arxiv.org/pdf/1604.07484.pdf eq. 3.2)
    where the kernel is treated as a sum of two kernels.

    k = [      k1, rho   * k1   + [0, 0
         rho * k1, rho^2 * k1]     0, k2]
    """

    def __init__(self,
                 num_tasks,
                 rank=1,  # for two multifidelity always assumed to be 1
                 prior=None,
                 includeParams=True,
                 **kwargs
                 ):
        if rank > num_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        super().__init__(**kwargs)
        try:
            self.batch_shape
        except AttributeError as e:
            self.batch_shape = 1  # torch.Size([200])

        # we take a power of rho with the task index list (assuming all task 0 represented as 0, task 1 represented as 1 etc.)
        self.covar_factor = torch.arange(num_tasks).to(torch.float32)

        if includeParams:
            self.register_parameter(name="rho", parameter=torch.nn.Parameter(torch.randn(1)))
            print(f"Initial value : rho  {self.rho.item()}")
            self.covar_factor = torch.pow(self.rho.repeat(num_tasks), self.covar_factor)

        self.covar_factor = self.covar_factor.unsqueeze(-1)
        #self.covar_factor = self.covar_factor.repeat(self.batch_shape, 1, 1)

        if prior is not None and includeParams is True:
            self.register_prior("rho_prior", prior, self._rho)

    def _rho(self):
        return self.rho

    def _eval_covar_matrix(self):
        transp = self.covar_factor.transpose(-1, 0)
        ret = self.covar_factor.matmul(self.covar_factor.transpose(-1, -2))  # + D
        return ret

    @property
    def covar_matrix(self):
        res = RootLinearOperator(self.covar_factor)
        print("root",res.to_dense())
        return res

    def forward(self, i1, i2, **params):
        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = torch.broadcast_shapes(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLinearOperator(
            base_linear_op=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class MultiFidelityGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultiFidelityGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # self.covar_module1 = gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.MaternKernel()
        #     )
        # self.covar_module2 = gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.MaternKernel()
        #     )
        self.covar_module1 =  gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=int(train_x[0].shape[1]))
        self.covar_module1.initialize_from_data(train_x[0], train_y)
        self.covar_module2 =  gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=int(train_x[0].shape[1]))
        self.covar_module2.initialize_from_data(train_x[0], train_y)

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        # self.task_covar_module = IndexKernel(num_tasks=2, rank=1)
        self.task_covar_module1 = TwoFidelityIndexKernel(num_tasks=2, rank=1)
        self.task_covar_module2 = TwoFidelityIndexKernel(num_tasks=2, rank=1,
                                                         includeParams=False)  # , batch_shape=(train_y.shape[0],1,1))
        #self.task_covar_module1 = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    #         print(self.covar_module1.outputscale.item())
    #         print(self.covar_module1.base_kernel.lengthscale.item())
    #         pprint(dir(self.covar_module1))
    #         pprint(dir(self.covar_module1.base_kernel))

    # print(f"Initial value : Covar 1, lengthscale {self.covar_module1.base_kernel.lengthscale.item()}, prefactor {self.covar_module1.outputscale.item()}")
    # print(f"Initial value : Covar 2, lengthscale {self.covar_module2.base_kernel.lengthscale.item()}, prefactor {self.covar_module2.outputscale.item()}")

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar1_x = self.covar_module1(x)
        # Get task-task covariance
        covar1_i = self.task_covar_module1(i)

        # Get input-input covariance
        covar2_x = self.covar_module2(x)
        # Get task-task covariance
        covar2_i = self.task_covar_module2(i)

        # Multiply the two together to get the covariance we want
        covar1 = covar1_x.mul(covar1_i)
        covar2 = covar2_x.mul(covar2_i)
        #         covar1 = covar1_x * covar1_i
        #         covar2 = covar2_x * covar2_i pipreqs ./ --encoding=utf8

        return gpytorch.distributions.MultivariateNormal(mean_x, covar1 + covar2)

"-----------------------PLOT FUNCTION-----------------------------------------------"
def plot_interplate(model, likelihood):
    i = np.linspace(0.1, 0.3, 7)
    x = np.linspace(0.1, 0.6, 6)
    y = np.linspace(5, 70, 14)
    z = np.linspace(0, 180, 19)
    I, X, Y, Z = np.meshgrid(i, x, y, z)
    I = I.flatten()
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    pointX = []
    for j in range(len(X)):
        px = np.array([I[j], X[j], Y[j], Z[j]])
        pointX.append(px)

    model.eval()
    likelihood.eval()
    pointX=np.array(pointX)
    K=torch.tensor(pointX).to(device).to(torch.float32)
    segment=np.linspace(0,len(X),10).astype(int)
    Y=np.array([])
    for i in range(9):
        A = likelihood(model(   K[segment[i]:segment[i+1],:]   )).mean.cpu().detach().numpy()
        Y=np.concatenate((Y,A))
    Y=np.expand_dims(Y,axis=1)
    Test=np.concatenate((pointX,Y),axis=1)
    np.savetxt("test.csv", Test, delimiter=',')
def plot3D(model, likelihood,num_task=1,scale=[1]*18):
    "num_task<0    ->use the multi-fidelity kernel"
    #num_task=1 Single GP; num_task=0 Raw;num_task=2 Multitask;num_task=-2  and multifidelity ;num_task=-1 OR num_task=1 MultiKernel=1 multifidelity
    #Multikernel means multifidelity kernel;
    #Raw means just plot the raw data
    model.eval()
    likelihood.eval()

    if num_task==0:
        Raw=1
        num_task=2
    else:
        Raw=0

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #for i in [0.1, 0.15, 0.2, 0.25, 0.4]:
        value = []
        for i in [0.1]:
            x = np.linspace(0.1, 0.6, 6)
            y = np.linspace(5, 40, 8)
            z = np.linspace(0, 180, 7)
            X, Y, Z = np.meshgrid(x, y, z)
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()

            pointx = []
            pointy = []
            for j in range(len(X)):
                px, py = findpoint_interpolate(np.array([i, X[j], Y[j], Z[j]]), Frame2,num_task,"nearest")
                # if np.isnan(py):
                #     px, py = findpoint_interpolate(np.array([i, X[j], Y[j], Z[j]]), Frame2, num_task,"nearest")

                pointx.append(px)
                pointy.append(py)
            pointX = np.asarray(pointx)
            values = np.asarray(pointy).T.squeeze(np.abs(num_task)-1)


            if np.abs(num_task)==1:
                value.append(values)

                if num_task<0:
                    values=likelihood(model( torch.tensor(pointX).to(torch.float32), torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=0))).mean
                    values0=likelihood(model( torch.tensor(pointX).to(torch.float32), torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=1))).mean
                    value.append(values0)
                else: values = likelihood(model(torch.tensor(pointX).to(torch.float32).to(device))).mean
                value.append(values)

            else:
                values=values.T
                value.append(values[:,0])
                value.append(values[:,1])

                if num_task==-2:
                    values = likelihood(*model(
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=0)),
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=0))
                    ))
                    value.append(values[0].mean)
                    value.append(values[1].mean)
                    values = likelihood(*model(
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=1)),
                        (torch.tensor(pointX).to(torch.float32),
                         torch.full((pointX.shape[0], 1), dtype=torch.long, fill_value=1))
                    ))

                    value.append(values[0].mean)
                    value.append(values[1].mean)
                    print("realx2,lowx2,highx2")
                elif num_task==2 and Raw==0:#independent multitask
                    values = likelihood(*model(
                        torch.tensor(pointX).to(torch.float32).to(device),
                        torch.tensor(pointX).to(torch.float32).to(device),

                    ))
                    value.append(values[0].mean.detach().cpu().numpy())
                    value.append(values[1].mean.detach().cpu().numpy())
                # elif Raw==0:
                #     value.append(likelihood(model(torch.tensor(pointX).to(torch.float32).to(device))).mean[:,0])
                #     value.append(likelihood(model(torch.tensor(pointX).to(torch.float32).to(device))).mean[:,1])

    # S = np.array([value[0], value[1].cpu().detach().numpy()]).T
    # Test=np.concatenate((pointX,S),axis=1)
    # np.savetxt("test.csv", Test, delimiter=',')
    scale2=[2,1.5,2,1.5,2,1.5,3,3,3,3,3,3]
    #scale2 = [1,1,3,3]
    for p in range(len(value)):
        fig = go.Figure(data=go.Isosurface(
            x=X,
            y=Y,
            z=Z,
            value=(value[p] * scale[p]).tolist(),
            isomin=-2 * scale[p] * scale2[p],
            # isomin=min(values)
            isomax=2 * scale[p] * scale2[p],
            # surface_fill=0.7,
            # opacity=0.9,  # 改变图形的透明度
            colorscale='jet',  # 改变颜色

            surface_count=5,
            colorbar_nticks=7,
            caps=dict(x_show=False, y_show=False, z_show=False),

            # slices_z = dict(show=True, locations=[-1, -9, -5]),
            # slices_y = dict(show=True, locations=[20]),

            # surface=dict(count=3, fill=0.7, pattern='odd'),  # pattern取值：'all', 'odd', 'even'
            # caps=dict(x_show=True, y_show=True),
            # surface_pattern = "even"
        ))
        fig.update_scenes(yaxis=dict(title=r'θ', tickfont=dict(size=13), titlefont=dict(size=18)))
        fig.update_scenes(yaxis_nticks=5)
        fig.update_scenes(xaxis_nticks=4)
        fig.update_scenes(xaxis_range=list([0, 0.7]))
        # fig.update_scenes(zaxis_nticks=3)
        fig.update_scenes(xaxis=dict(title=r'y', tickfont=dict(size=13), titlefont=dict(size=18)))
        fig.update_scenes(zaxis=dict(title='ψ', tickfont=dict(size=13), titlefont=dict(size=18)))
        fig.update_coloraxes(colorbar_tickfont_size=20)
        fig.update_layout(
            height=500,
            width=500,
        )
        fig.show()
        #pio.write_image(fig, f'3D_True_{x[0,0]}_{j}eposide.png')


def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    return im


def plot_2d(observed_pred_y1, observed_pred_y2, test_y_actual1, test_y_actual2, delta_y1, delta_y2):
    # Plot our predictive means
    f, observed_ax = plt.subplots(2, 3, figsize=(4, 3))
    ax_plot(f, observed_ax[0, 0], observed_pred_y1, 'observed_pred_y1 (Likelihood)')

    # Plot the true values
    # f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
    ax_plot(f, observed_ax[1, 0], observed_pred_y2, 'observed_pred_y2 (Likelihood)')

    # Plot the absolute errors
    ax_plot(f, observed_ax[0, 1], test_y_actual1, 'test_y_actual1')

    # Plot the absolute errors
    ax_plot(f, observed_ax[1, 1], test_y_actual2, 'test_y_actual2')

    # Plot the absolute errors
    ax_plot(f, observed_ax[0, 2], delta_y1, 'Absolute Error Surface1')

    # Plot the absolute errors
    im = ax_plot(f, observed_ax[1, 2], delta_y2, 'Absolute Error Surface2')

    cb_ax = f.add_axes([0.9, 0.1, 0.02, 0.8])  # 设置colarbar位置
    cbar = f.colorbar(im, cax=cb_ax)  # 共享colorbar

    plt.show()
#________________________________________GA
def evaluateMT(individual,model,likelihood):
    model.eval()
    likelihood.eval()
    ind=[0]*len(STP)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(len(STP)):
            ind[i]=individual[i]*(UPB[i]-LOWB[i])/(STP[i]-1)+LOWB[i]
        ind=torch.tensor(ind).to(device).to(torch.float32).unsqueeze(0)
        A=likelihood(*model( ind,  ind))
    return   A[0].mean.item(),A[1].mean.item()
def feasibleMT(ind):
    # 判定解是否满足约束条件
    # 如果满足约束条件，返回True，否则返回False
    for i in range(len(STP)) :
        if   (STP[i]-ind[i])<=0:
            return False
    return  True
def evaluateEI(individual,model,likelihood,y_max,cofactor):
    model.eval()
    likelihood.eval()
    ind=[0]*len(STP)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(len(STP)):
            ind[i]=individual[i]*(UPB[i]-LOWB[i])/(STP[i]-1)+LOWB[i]
        ind=torch.tensor(ind).to(device).to(torch.float32).unsqueeze(0)
        A=likelihood(*model( ind,  ind))

        VarS = A[0].variance
        MeanS = A[0].mean
        VarS2 = A[1].variance
        MeanS2 = A[1].mean

        VarS = VarS + 1e-05  # prevent var=0
        EI_one = (MeanS - y_max[0]) * torch.FloatTensor(norm.cdf(((MeanS - y_max[0]) / VarS).cpu().detach())).to(device)
        EI_two = VarS * torch.FloatTensor(norm.pdf((MeanS - y_max[0] / VarS).cpu().detach())).to(device)
        #EI = EI_one * cofactor[0] + EI_two * (1 - cofactor[0])

        VarS2 = VarS2 + 1e-05
        EI_one1 = (MeanS2 - y_max[1]) * torch.FloatTensor(norm.cdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(
            device)
        EI_two1 = VarS2 * torch.FloatTensor(norm.pdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(device)
        #EI1 = EI_one1 * cofactor[0] + EI_two1 * (1 - cofactor[0])

    return  ( EI_one*cofactor[1]+EI_one1).item(),(EI_two*(1-cofactor[1])+EI_two1).item()

def evaluateEISO(individual,model,likelihood,y_max,cofactor):
    model.eval()
    likelihood.eval()
    ind=[0]*len(STP)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(len(STP)):
            ind[i]=individual[i]*(UPB[i]-LOWB[i])/(STP[i]-1)+LOWB[i]
        ind=torch.tensor(ind).to(device).to(torch.float32).unsqueeze(0)
        A=likelihood(*model( ind,  ind))

        VarS = A[0].variance
        MeanS = A[0].mean
        VarS2 = A[1].variance
        MeanS2 = A[1].mean

        VarS = VarS + 1e-05  # prevent var=0
        EI_one = (MeanS - y_max[0]) * torch.FloatTensor(norm.cdf(((MeanS - y_max[0]) / VarS).cpu().detach())).to(device)
        EI_two = VarS * torch.FloatTensor(norm.pdf((MeanS - y_max[0] / VarS).cpu().detach())).to(device)
        EI = EI_one * cofactor[0] + EI_two * (1 - cofactor[0])

        VarS2 = VarS2 + 1e-05
        EI_one1 = (MeanS2 - y_max[1]) * torch.FloatTensor(norm.cdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(
            device)
        EI_two1 = VarS2 * torch.FloatTensor(norm.pdf(((MeanS2 - y_max[1]) / VarS2).cpu().detach())).to(device)
        EI1 = EI_one1 * cofactor[0] + EI_two1 * (1 - cofactor[0])

    return   EI.item(),EI1.item()
if __name__=="__main__":
    pass