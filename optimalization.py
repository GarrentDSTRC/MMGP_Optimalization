#全局优化
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
pathpop=r".\Database\pop.npy"
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
    #full_train_y[:,1] = 10*full_train_y[:,1]
    if os.path.exists(path4):
        dict = np.load(path4, allow_pickle=True).astype(int).tolist()

# Here we have two iterms that we're passing in as train_inputs
likelihood1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
#50: 0:2565
model1 = MultiFidelityGPModel((full_train_x[:,:], full_train_i[:,:]), full_train_y[:,0], likelihood1).to(device)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model2 = MultiFidelityGPModel((full_train_x[:,:], full_train_i[:,:]), full_train_y[:,1], likelihood2).to(device)
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
model.eval()
likelihood.eval()



# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from pymoo.factory import get_problem, get_reference_directions
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.population import Population


# Define a custom problem class that takes the observed predictions as objectives
class MyProblem(Problem):

    def __init__(self, ):
        super().__init__(n_var=7,
                         n_obj=2,
                         n_constr=1,
                         xu=np.array([1.0,0.6,40,180,9,9,35]),
                         xl=np.array([0.6,0.1,5,0,0,0,10]))


    def _evaluate(self, x, out, *args, **kwargs):
        # Use the observed predictions as objectives
        test_x = torch.tensor(x).to(torch.float32)
        test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)

        # Make predictions - one task at a time
        # We control the task we cae about using the indices

        # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
        # See https://arxiv.org/abs/1803.06058
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_yH = likelihood(*model((test_x, test_i_task2), (test_x, test_i_task2)))
            observed_pred_yHC = -1*np.array(observed_pred_yH[0].mean.tolist())  # ct high
            observed_pred_yHE = -1*np.array(observed_pred_yH[1].mean.tolist())  # eta high


        N=np.array([observed_pred_yHC,observed_pred_yHE]).T
        #print(N)
        out["F"] =N
        #N1=-N[:, 1]-0.97
        out["G"] = [N[:, 0]]         #.reshape(-1, 3, 1)
        #out["G"] = N1.reshape(-1, 1)
# # Create an instance of the problem with the observed predictions
problem = MyProblem()
#
# # Define the reference directions for the Pareto front
from pymoo.util.ref_dirs import get_reference_directions
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

# Initialize the algorithm with the last 150 samples from full_train_x
pop = Population.new("X", full_train_x[-2000:,:].numpy())
#pop.set("F", problem.evaluate(pop.get("X")))
# Create an instance of the NSGA-II algorithm
algorithm = NSGA2(pop_size=300, eliminate_duplicates=True,sampling=pop)

# Minimize the problem using the algorithm and the initial population
res = minimize(problem,
               algorithm,
               ("n_gen", 10),#70
               seed=3,
)
# Get the last population from the result object
last_pop = res.pop

# Save the last population to the path
np.save(pathpop, last_pop)


plt.scatter(-1*res.F[:, 0], -1*res.F[:, 1], c="blue", marker="o", s=20) # Make the markers smaller
plt.xlabel("ct ", fontsize=22, fontfamily='Times New Roman') # Increase the font size and use Times New Roman font
plt.ylabel("cl ", fontsize=22, rotation=0, fontfamily='Times New Roman') # Rotate the y-axis label to be vertical
#plt.title("Pareto front of ct and η for different designs", fontsize=18, fontfamily='Times New Roman') # Increase the font size of the title
plt.rc('font', family='Times New Roman') # Set the font family for all text elements

# Add a legend with larger font size
#plt.legend(["Designs"], loc="upper right", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
# Save the plot as a high-resolution PDF file
plt.subplots_adjust(bottom=0.2)
plt.savefig("pareto_front.pdf", dpi=300)

# Show the plot on screen
plt.show()


# Import the pandas library for data manipulation
import pandas as pd

# Get the Pareto front solutions from the result object
pf = res.opt

# Get the decision variables and objective values of the Pareto front solutions
X = pf.get("X")
F = -pf.get("F")

# Create a pandas dataframe with the decision variables and objective values
df = pd.DataFrame(np.hstack([X, F]), columns=["st",	"ad","theta","phi","m","p","t", "ct", "cl"])

# Save the dataframe to a csv file
df.to_csv("pareto_front.csv", index=False)