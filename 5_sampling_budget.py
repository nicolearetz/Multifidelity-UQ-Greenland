## Imports
import numpy as np
import pickle
import time

import sys
sys.path.insert(0, '../../source')
from ModelManagerPrecomputed import BlueOverPrecomputed

# user settings
example_no = 0
target_budget_factor = 2

# initializations (so that we can save in-between)
result_mfmc = None
overhead_mfmc = None
result_mlmc = None
overhead_mlmc = None
mlblue_result = None
overhead_mlblue = None

# load data
filename = "covariance"
with open(filename, "rb") as file:
    covariances, correlations, dV, rescaling, mean_costs_CPU = pickle.load(file)
n_models = covariances[0].shape[0]

# initializations
precomputed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
mfmc_data = None
mlmc_data = None
mlblue_data = None
target_budget = 5 * mean_costs_CPU[0]

my_options = {
    "feastol": 1e-7,
    "abstol": 1e-8,
    "reltol": 1e-5
}

# load data
filename = "data/precomputed_samples"
with open(filename, "rb") as output_file:
    loaded = pickle.load(output_file)

outputs_precomputed = loaded[0]
costs_precomputed = loaded[1]
parameters = loaded[2]
reference_cost = loaded[9]

n_samples = parameters.shape[0]

## MFMC
tStart = time.time()

# set up sampler class
sampler = BlueOverPrecomputed(n_models,
                              costs=reference_cost,
                              mlmc_variances=dV.tolist(),
                              C=covariances.tolist(),
                              skip_projection=False,
                              n_outputs=35)
sampler.base_setup(n_models=n_models, index2sample=np.arange(n_samples))

for i in precomputed_list:
    sampler.set_sample(i, outputs_precomputed[i])

# adjust input data to account for many outputs
mfmc_data = sampler.setup_mfmc(budget=target_budget, continuous_relaxation=False, small_budget=True)

# save predictions
filename = "MF_estimates_budget"
with open(filename, "wb") as output_file:
    pickle.dump([result_mfmc, overhead_mfmc, mfmc_data,
                 result_mlmc, overhead_mlmc, mlmc_data,
                 mlblue_result, overhead_mlblue, mlblue_data,
                 costs_precomputed, covariances],
                output_file)

result_mfmc = sampler.solve_mfmc(mfmc_data=mfmc_data)
overhead_mfmc = time.time()-tStart

# save predictions
with open(filename, "wb") as output_file:
    pickle.dump([result_mfmc, overhead_mfmc, mfmc_data,
                 result_mlmc, overhead_mlmc, mlmc_data,
                 mlblue_result, overhead_mlblue, mlblue_data,
                 costs_precomputed, covariances],
                output_file)

## MLMC
tStart = time.time()

# set up sampler class
sampler = BlueOverPrecomputed(n_models,
                              costs=reference_cost,
                              mlmc_variances=dV.tolist(),
                              C=covariances.tolist(),
                              skip_projection=False,
                              n_outputs=35)
sampler.base_setup(n_models=n_models, index2sample=np.arange(n_samples))

for i in precomputed_list:
    sampler.set_sample(i, outputs_precomputed[i])

mlmc_data = sampler.setup_mlmc(budget=target_budget, continuous_relaxation=False, small_budget=True)

# save predictions
with open(filename, "wb") as output_file:
    pickle.dump([result_mfmc, overhead_mfmc, mfmc_data,
                 result_mlmc, overhead_mlmc, mlmc_data,
                 mlblue_result, overhead_mlblue, mlblue_data,
                 costs_precomputed, covariances],
                output_file)

result_mlmc = sampler.solve_mlmc(mlmc_data=mlmc_data)
overhead_mlmc = time.time()-tStart

# save predictions
with open(filename, "wb") as output_file:
    pickle.dump([result_mfmc, overhead_mfmc, mfmc_data,
                 result_mlmc, overhead_mlmc, mlmc_data,
                 mlblue_result, overhead_mlblue, mlblue_data,
                 costs_precomputed, covariances],
                output_file)

## MLBLUE
tStart = time.time()
problem = BlueOverPrecomputed(n_models,
                             costs=mean_costs_CPU,
                             C=covariances.tolist(),
                             n_outputs=35,
                             mlmc_variances=dV.tolist(),
                             skip_projection=False)

mlblue_data = problem.setup_solver(K=5, budget=target_budget, optimization_solver_params=my_options)
n_samples = parameters.shape[0]

# save predictions
with open(filename, "wb") as output_file:
    pickle.dump([result_mfmc, overhead_mfmc, mfmc_data,
                 result_mlmc, overhead_mlmc, mlmc_data,
                 mlblue_result, overhead_mlblue, mlblue_data,
                 costs_precomputed, covariances],
                output_file)

# get the index-to-parameter map
sample_assignment = -1 * np.ones((n_samples, n_models), dtype = int)
stop = 0

for m, modelset in enumerate(mlblue_data['models']):
    for model in modelset:
        sample_assignment[stop:stop+mlblue_data['samples'][m], model] = [*range(stop, stop+mlblue_data['samples'][m])]
    stop = stop+mlblue_data['samples'][m]

index2para = np.arange(n_samples, dtype=int)

stop = 0
for i in precomputed_list[:-1]:
    where = sample_assignment[np.where(sample_assignment[:, i] > -0.5)[0], i]
    n_swaps = where.shape[0]
    merk = index2para[stop:stop+n_swaps].copy()
    index2para[stop:stop+n_swaps] = sample_assignment[where, i]
    index2para[where] = merk
    sample_assignment[where, :] = -1
    stop = stop + n_swaps

problem.base_setup(n_models=n_models, index2sample=index2para, output_numbers=[*range(35)])
for i in precomputed_list:
    problem.set_sample(i, outputs_precomputed[i])

mlblue_result = problem.solve(eps=problem.MOSAP_output['eps'], budget=problem.MOSAP_output['budget'])
overhead_mlblue = time.time()-tStart


# save predictions
with open(filename, "wb") as output_file:
    pickle.dump([result_mfmc, overhead_mfmc, mfmc_data,
                 result_mlmc, overhead_mlmc, mlmc_data,
                 mlblue_result, overhead_mlblue, mlblue_data,
                 costs_precomputed, covariances],
                output_file)
