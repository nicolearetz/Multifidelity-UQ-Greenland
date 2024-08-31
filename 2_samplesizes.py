## Imports
import numpy as np
import pickle
import time
import bluest

# in this script, we choose the sample sizes for a computational budget to approximate the expected ice mass change
# for the final time t=2050.
# run with command
# python3 samplesizes.py

# script settings
n_test = 251

# load data
filename = "covariance".format()
with open(filename, "rb") as file:
    covariance, correlation, dV, rescaling, mean_costs = pickle.load(file)
n_models = covariance[0].shape[0]

# restrict to t=2050 only
covariances = covariance[[-1]]
correlations = correlation[[-1]]
dV = dV[[-1]]

# initializations
budgets = mean_costs[0] * np.logspace(0, 2.5, n_test)
errors = np.zeros((3, n_test))
cost_opt = np.zeros((3, n_test))
chosen = np.zeros((3, n_test), dtype=object)
samplesizes = np.zeros((3, n_test), dtype=object)
results = np.zeros((3, n_test), dtype=object)
sample_allocation_cost = np.zeros((3, n_test))

## MFMC
i = 0
# set up bluest problem class
problem = bluest.BLUEProblem(n_models,
                             costs=mean_costs,
                             C=covariances.tolist(),
                             n_outputs=1,
                             mlmc_variances=dV.tolist(),
                             skip_projection=False)

for j in range(1, n_test):
    tStart = time.time()
    MFMC_data = problem.setup_mfmc(budget=budgets[j], continuous_relaxation=False, small_budget=True)

    results[i,j] = MFMC_data
    errors[i, j] = MFMC_data['errors'][0]
    cost_opt[i,j] = MFMC_data['total_cost']
    chosen[i,j] = MFMC_data['models']
    samplesizes[i,j] = MFMC_data['samples']
    sample_allocation_cost[i,j] = time.time()-tStart

## MLMC
i = 1
# set up bluest problem class
problem = bluest.BLUEProblem(n_models,
                             costs=mean_costs,
                             C=covariances.tolist(),
                             n_outputs=1,
                             mlmc_variances=dV.tolist(),
                             skip_projection=False)

for j in range(1, n_test):
    tStart = time.time()

    MLMC_data = problem.setup_mlmc(budget=budgets[j], continuous_relaxation=False, small_budget=True)

    results[i, j] = MLMC_data
    errors[i, j] = MLMC_data['errors'][0]
    cost_opt[i, j] = MLMC_data['total_cost']
    chosen[i, j] = MLMC_data['models']
    samplesizes[i, j] = MLMC_data['samples']

    sample_allocation_cost[i, j] = time.time() - tStart

## MLBLUE
i = 2
my_options = {
    "feastol": 1e-7,
    "abstol": 1e-8,
    "reltol": 1e-5
}
blue_sol_for_budget = np.arange(n_test)

problem = bluest.BLUEProblem(n_models,
                             costs=mean_costs,
                             C=covariances.tolist(),
                             n_outputs=1,
                             mlmc_variances=dV.tolist(),
                             skip_projection=False)

errors[i, 0] = np.sqrt(covariances[0][0,0] * 1 / np.floor(budgets[0]/mean_costs[0]))

tStart_outer = time.time()
for j in range(1, n_test):
    tStart = time.time()

    try:
        MLMC_data = problem.setup_solver(K=5, budget=budgets[j], optimization_solver_params=my_options)

        if j > 0 and MLMC_data['errors'][0] > errors[i, j - 1]:
            results[i, j] = results[i, j - 1]
            errors[i, j] = errors[i, j - 1]
            cost_opt[i, j] = cost_opt[i, j - 1]
            chosen[i, j] = chosen[i, j - 1]
            samplesizes[i, j] = samplesizes[i, j - 1]
            blue_sol_for_budget[j] = blue_sol_for_budget[j - 1]
        else:
            results[i, j] = MLMC_data
            errors[i, j] = MLMC_data['errors'][0]
            cost_opt[i, j] = MLMC_data['total_cost']
            chosen[i, j] = MLMC_data['models']
            samplesizes[i, j] = MLMC_data['samples']

    except:
        print("Something went wrong with the minimization")
        errors[i, j] = np.nan
        cost_opt[i, j] = np.nan
        chosen[i, j] = np.nan
        samplesizes[i, j] = np.nan

    sample_allocation_cost[i, j] = time.time() - tStart

    print("Runtime of this block until now: {} min".format((time.time() - tStart_outer) / 60))

    ## saving
    filename = "samplesizes_t2050"
    with open(filename, "wb") as output_file:
        pickle.dump([rescaling,
                     mean_costs,
                     sample_allocation_cost,
                     results,
                     samplesizes,
                     chosen,
                     cost_opt,
                     errors,
                     budgets,
                     blue_sol_for_budget,
                     covariances], output_file)