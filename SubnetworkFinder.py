import pandas as pd


class SubnetworkFinder:
    """
    Class to detect functional networks in specific regions

    Attributes:

    Methods:

    """

    def find_network_by_linear_correlation(self, activity_matrix):
        """
        Method to find the neurons in this area which are correlated in
        terms of activity - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3586814/

        'Measure of degree to which trial-to-trial fluctuations in response strength
        are shared by a pair of neurons - Pearson correlation of their spike count responses'

        Population Pearson correlation coefficient
        cov(X, Y)/(stdX*sddY), where the X and Y are the inner lists of spike times

        :param activity_matrix:
        [
            [list of spike counts per trial] #Neuron 1 - each trial is a feature
            [list of spike counts per trial] #Neuron 2
            .
            .
            .
            [list of spike counts per trial] #Neuron n
        ]

        :return: adjacency_matrix: result of pair-wise correlations between their activities

        Matrix of dimensions: number_neurons x number_neurons - length of activity_matrix
        """
        adjacency_matrix = activity_matrix.corr()

        return adjacency_matrix


"""
    Drives the SubnetworkFinder during development
"""
"""if __name__ == "__main__":
    # Dummy spike count matrix for 3 neurons with 10 trials
    spikes_per_trial_neuron_x = pd.Series(range(10, 20))
    spikes_per_trial_neuron_y = pd.Series([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])
    spikes_per_trial_neuron_z = pd.Series([5, 3, 2, 1, 0, 7, 0, 0, 1, 0])

    spike_count_matrix = pd.DataFrame({'neuron_x': spikes_per_trial_neuron_x,
                                       'neuron_y': spikes_per_trial_neuron_y,
                                       'neuron_z': spikes_per_trial_neuron_z})

    SubnetworkFinder().find_network_by_linear_correlation(spike_count_matrix)"""