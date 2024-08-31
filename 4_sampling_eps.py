## Imports
import numpy as np
import pickle
import time
import random

from ModelManagerPrecomputed import BlueOverPrecomputed

n_samples = 500000
stops = [0, 110, 600 + 110, 1800 + 600 + 110, n_samples]

def my_shuffle(outputs):
    order = [*range(n_samples)]
    for i in range(1, 5):
        yolo = order[stops[i - 1]:stops[i]]
        randomizer.shuffle(yolo)
        # np.random.shuffle(yolo)
        order[stops[i - 1]:stops[i]] = yolo

    for i in range(outputs.shape[0]):
        if not isinstance(outputs[i], int):
            outputs[i] = outputs[i][order[:outputs[i].shape[0]], :]

    return outputs

# user settings
target_accuracy_mm = 0.51
n_repeats = 10
seed = 804

filename = "covariance"
with open(filename, "rb") as file:
    covariances, correlations, dV, rescaling, mean_costs_CPU = pickle.load(file)
n_models = covariances[0].shape[0]

precomputed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
target_accuracy = 361.8 * target_accuracy_mm / rescaling

filename = "chosen_MF_setup"
with open(filename, "rb") as output_file:
    loaded = pickle.load(output_file)

mfmc_data = loaded[0]
mlmc_data = loaded[1]
mlblue_data = loaded[2]

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

result_mfmc = np.zeros(n_repeats, dtype = object)
result_mlmc = np.zeros(n_repeats, dtype = object)
result_mlblue = np.zeros(n_repeats, dtype = object)

## MFMC
tStart = time.time()

randomizer = random.Random(seed)
for j in range(n_repeats):
    # set up sampler class
    sampler = BlueOverPrecomputed(n_models,
                                  costs=reference_cost,
                                  mlmc_variances=dV.tolist(),
                                  C=covariances.tolist(),
                                  skip_projection=False,
                                  n_outputs=35)
    sampler.base_setup(n_models=n_models, index2sample=np.arange(n_samples))

    outputs_shuffeled = my_shuffle(outputs_precomputed.copy())

    for i in precomputed_list:
        sampler.set_sample(i, outputs_shuffeled[i])

    # adjust input data to account for many outputs
    mfmc_data = sampler.compute_mfmc_data(mfmc_data['models'], mfmc_data['samples'], bool_ignore_feasibility=True)
    result_mfmc[j] = sampler.solve_mfmc(mfmc_data=mfmc_data)


overhead_mfmc = time.time()-tStart


## MLMC
tStart = time.time()

randomizer = random.Random(seed)
for j in range(n_repeats):
    # set up sampler class
    sampler = BlueOverPrecomputed(n_models,
                                  costs=reference_cost,
                                  mlmc_variances=dV.tolist(),
                                  C=covariances.tolist(),
                                  skip_projection=False,
                                  n_outputs=35)
    sampler.base_setup(n_models=n_models, index2sample=np.arange(n_samples))

    outputs_shuffeled = my_shuffle(outputs_precomputed.copy())
    for i in precomputed_list:
        sampler.set_sample(i, outputs_shuffeled[i])

    mlmc_data = sampler.compute_mlmc_data(mlmc_data['models'], mlmc_data['samples'])
    result_mlmc[j] = sampler.solve_mlmc(mlmc_data=mlmc_data)
overhead_mlmc = time.time()-tStart

## MLBLUE
tStart = time.time()

randomizer = random.Random(seed)
for j in range(n_repeats):
    print("iteration ", j, " of ", n_repeats)

    problem = BlueOverPrecomputed(n_models,
                                 costs=mean_costs_CPU,
                                 C=covariances.tolist(),
                                 n_outputs=35,
                                 mlmc_variances=dV.tolist(),
                                 skip_projection=False)

    mlblue_data = problem.setup_solver(K=5, eps=target_accuracy, optimization_solver_params=my_options, bool_skip_solve=True, MLBLUE_data=mlblue_data)
    n_samples = parameters.shape[0]


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

    outputs_shuffeled = my_shuffle(outputs_precomputed.copy())
    problem.base_setup(n_models=n_models, index2sample=index2para, output_numbers=[*range(35)])
    for i in precomputed_list:
        problem.set_sample(i, outputs_shuffeled[i])

    result_mlblue[j] = problem.solve(eps=problem.MOSAP_output['eps'], budget=problem.MOSAP_output['budget'])
overhead_mlblue = time.time()-tStart


# save predictions
filename = "MF_estimates_eps"
with open(filename, "wb") as output_file:
    pickle.dump([result_mfmc, overhead_mfmc, mfmc_data,
                 result_mlmc, overhead_mlmc, mlmc_data,
                 result_mlblue, overhead_mlblue, mlblue_data,
                 costs_precomputed, covariances],
                output_file)

