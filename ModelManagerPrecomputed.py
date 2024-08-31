import numpy as np
from bluest import BLUEProblem

class BlueOverPrecomputed(BLUEProblem):

    n_outputs = 35

    def base_setup(self, n_models, index2sample, output_numbers=None):
        self.n_models = n_models
        self.index2sample = index2sample
        self.precomputed_samples = np.zeros(n_models, dtype=object)
        self.costs = np.zeros(n_models, dtype=object)
        self.current_sample_index = 0
        self.sample_combinations = np.zeros(n_models, dtype=object)

        if output_numbers is None:
            self.output_numbers = [*range(self.n_outputs)]
        else:
            self.output_numbers = output_numbers
    def sampler(self, ls, N2):
        samples = [[self.current_sample_index] for i in range(len(ls))]
        self.current_sample_index += 1
        return samples

    def evaluate(self, ls, samples):

        L = len(ls)
        out = [[0 for i in range(L)] for n in range(self.n_outputs)]

        for i in range(L):

            sample = samples[i].copy()
            model_number = ls[i]

            output = self.precomputed_samples[model_number][self.index2sample[sample][0], self.output_numbers]
            self.sample_combinations[model_number].append(sample[0])

            for n in range(self.n_outputs):
                # recall that samples[ls[i]] contains the input parameters for model ls[i]
                # out[n][i] will thus contain the n-th output of the model model_number given the input samples[model_number]
                out[n][i] = output[n]  # to get everything in the format that Matteo needs

        return out

    def set_sample(self, model_no, samples):
        self.precomputed_samples[model_no] = samples
        self.sample_combinations[model_no] = []

