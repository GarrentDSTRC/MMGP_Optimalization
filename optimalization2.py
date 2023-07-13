#双层GA
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
training_iterations = 13#33#5
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
    last_pop=np.load(pathpop, allow_pickle=True)
    # Get the F and X values from the population
    F = np.array([ind.get("F") for ind in last_pop])
    X = np.array([ind.get("X") for ind in last_pop])
    # Sum the two dimensions of F
    F_sum = F.sum(axis=1)
    # Sort the F_sum and X by ascending order of F_sum
    sorted_indices = np.argsort(F_sum)
    F_sum_sorted = F_sum[sorted_indices]
    X_sorted = X[sorted_indices]
    # Get the X numpy array
    pop = torch.FloatTensor(X_sorted)
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

class InnerProblem(Problem):
    def __init__(self, outer_variables):
        super().__init__(n_var=4, n_obj=2, n_constr=1, xu=np.array([1.0,0.6,40,180]), xl=np.array([0.6,0.1,5,0]))
        self.outer_variables = outer_variables

    def _evaluate(self, x, out, *args, **kwargs):
        # Join the inner and outer variables
        full_variables = np.concatenate([x, np.repeat(self.outer_variables[None, :], len(x), axis=0)], axis=1)

        # Use these variables as inputs to your GP model
        test_x = torch.tensor(full_variables).to(torch.float32)
        test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_yH = likelihood(*model((test_x, test_i_task2), (test_x, test_i_task2)))
            observed_pred_yHC = -1 * np.array(observed_pred_yH[0].mean.tolist())  # ct high
            observed_pred_yHE = - np.array(observed_pred_yH[1].mean.tolist()) # eta high
        N=np.array([observed_pred_yHC,observed_pred_yHE]).T
        #print(N)
        out["F"] =N
        #N1=-N[:, 1]-0.97
        out["G"] = [N[:, 0]]
        # I've omitted the constraints for simplicity


class OuterProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xu=np.array([9,9,35]), xl=np.array([0,0,10]))
#,type_var=np.int
    def _evaluate(self, x, out, *args, **kwargs):
        results = []
        #x = np.round(x).astype(int)  # Convert the variables to integers
        for variables in x:
            problem = InnerProblem(variables)
            pop_2 = Population.new("X", pop[:, :4].numpy())
            algorithm = NSGA2( pop_size=700, eliminate_duplicates=True,sampling=pop_2)
            res = minimize(problem, algorithm, termination=("n_gen", 15),seed=1)
            max_values = np.min(res.F, axis=0)  # Get the maximum value for each objective
            print(max_values)
            results.append(max_values)
        out["F"] = np.array(results)


# Run the outer GA
problem = OuterProblem()
pop_1 = Population.new("X", np.concatenate((pop[0:3,4:].numpy(), pop[3:15, 4:].numpy())))
algorithm = NSGA2(pop_size=600, eliminate_duplicates=True,sampling=pop_1)
res = minimize(problem, algorithm, termination=("n_gen", 5),seed=1)

plt.scatter(-1*res.F[:, 0], -1*res.F[:, 1], c="blue", marker="o", s=20) # Make the markers smaller
plt.xlabel("f1 ", fontsize=22, fontfamily='Times New Roman') # Increase the font size and use Times New Roman font
plt.ylabel("f2", fontsize=22, rotation=0, fontfamily='Times New Roman') # Rotate the y-axis label to be vertical
#plt.title("Pareto front of ct and η for different designs", fontsize=18, fontfamily='Times New Roman') # Increase the font size of the title
plt.rc('font', family='Times New Roman') # Set the font family for all text elements

# Add a legend with larger font size
#plt.legend(["Designs"], loc="upper right", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
# Save the plot as a high-resolution PDF file
plt.subplots_adjust(bottom=0.2)
plt.savefig("pareto_frontONE.pdf", dpi=300)

# Show the plot on screen
plt.show()


# Import the pandas library for data manipulation
import pandas as pd

# Get the Pareto front solutions from the result object
pf = res.opt

# Get the decision variables and objective values of the Pareto front solutions
X = pf.get("X")
F = -pf.get("F")
print(X,F)
# Create a pandas dataframe with the decision variables and objective values
df = pd.DataFrame(np.hstack([X, F]), columns=["m","p","t", "ct", "cl"])

# Save the dataframe to a csv file
df.to_csv("pareto_frontONE.csv", index=False)