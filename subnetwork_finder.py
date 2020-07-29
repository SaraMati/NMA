import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
from scipy import stats
import math


class SubnetworkFinder:
    """
    Class to detect and visualise functional networks in specific regions
    """
    @staticmethod
    def above_percentile(cell_value, flattened_matrix, percentile):
        if math.isnan(cell_value):
            return False
        return stats.percentileofscore(flattened_matrix, cell_value) >= percentile

    @staticmethod
    def find_functional_subnetwork(adjacency_matrix, percentile=75):
        upper_half = SubnetworkFinder.get_upper_half_of_adjacency_matrix(adjacency_matrix)

        flattened_matrix = upper_half.to_numpy().flatten()
        flattened_matrix = flattened_matrix[~np.isnan(flattened_matrix)]

        transformed_upper_half = upper_half.applymap(
            lambda cell: SubnetworkFinder.above_percentile(cell, flattened_matrix, percentile))

        np.fill_diagonal(transformed_upper_half.values, False)

        # transformed_upper_half now has true when there is an 'above-threshold' connection
        # between neurons
        return transformed_upper_half.columns[(transformed_upper_half == True).any(axis=1)], transformed_upper_half

    @staticmethod
    def create_graph_of_network(thresholded_adjacency_matrix, include_below_threshold):
        """
        Given the thresholded adjacency matrix, transforms into networkx and removes isolates
        :param thresholded_adjacency_matrix:
        :param include_below_threshold:
        :return: networkx graph
        """
        if not include_below_threshold:
            thresholded_adjacency_matrix = \
                thresholded_adjacency_matrix[(thresholded_adjacency_matrix == True).any(axis=1)]

            # Create the graph without isolates
        g = nx.from_pandas_adjacency(thresholded_adjacency_matrix)
        g.remove_nodes_from(list(nx.isolates(g)))
        return g

    @staticmethod
    def get_upper_half_of_adjacency_matrix(adjacency_matrix):
        mask = np.triu(np.ones_like(adjacency_matrix, dtype=np.bool))
        upper_half = adjacency_matrix.where(mask).dropna(axis='columns', how='all').dropna(axis='rows', how='all')
        return upper_half

    @staticmethod
    def find_adjacency_matrix_from_shuffled_data(activity_matrix):
        activity_matrix = activity_matrix.apply(lambda x: np.random.shuffle(x) or x, axis=1)
        baseline_adjacency_matrix = SubnetworkFinder.find_network_by_linear_correlation_full_window(activity_matrix)
        return baseline_adjacency_matrix

    @staticmethod
    def find_network_by_linear_correlation_full_window(activity_matrix):
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

        adjacency_matrix = activity_matrix.T.corr()

        return adjacency_matrix


class SubnetworkVisualiser:
    """
    Class to create visualisations for a region
    """
    def __init__(self, region):
        self.region = region

    def create_heatmap_from_adjacency_matrix(self, adjacency_matrix):
        sns.heatmap(adjacency_matrix,
                    fmt='.1g',
                    vmin=-1, vmax=1, center=0,
                    cmap='coolwarm',
                    yticklabels=True, xticklabels=True).set_title(self.region)
        plt.show()

    def create_histogram_of_correlations(self, adjacency_matrix, percentile):
        """
        Plot histogram to show distribution of the correlations
        Each pair will be counted once (i.e. Neuron1-Neuron2 counted,
        Neuron2-Neuron1 is not counted)

        Should help inform the decision on a threshold

        :param adjacency_matrix:
        :param percentile: use this to display a line of where the percentile is falling
        :return:
        """
        upper_half = SubnetworkFinder.get_upper_half_of_adjacency_matrix(adjacency_matrix)
        all_pearson_values = upper_half.to_numpy().flatten()
        all_pearson_values = all_pearson_values[~np.isnan(all_pearson_values)]

        point_for_vertical_line = np.quantile(all_pearson_values, percentile/100)

        plot = sns.distplot(all_pearson_values)
        plot.set(title=self.region, xlabel='Pearson Correlation Coefficient', ylabel='Count')
        plot.axvline(point_for_vertical_line, color='red')
        plot.text(point_for_vertical_line, 0, str(percentile) + ' percentile', rotation=90)
        plt.show()

    def create_histogram_of_cell_types(self, cells_with_extra_info):
        histogram = sns.catplot(x="Cell_type", kind="count", palette="ch:.25", data=cells_with_extra_info)
        histogram.fig.suptitle(self.region)
        plt.show()

    def create_graph_diagram(self, thresholded_adjacency_matrix, cells_with_type, include_below_threshold=True):

        g = SubnetworkFinder.create_graph_of_network(thresholded_adjacency_matrix, include_below_threshold)
        # TODO: shouldn't be hardcoded but just generate a map of colours
        # for the unique values in cells_with_type
        cell_types_per_neuron = cells_with_type['Cell_type']
        colour_map = []
        for neuron_index in g.nodes():
            try:
                cell_type = cell_types_per_neuron[neuron_index]
                if cell_type == 'Narrow Interneuron':
                    colour_map.append('blue')
                elif cell_type == 'Wide Interneuron':
                    colour_map.append('green')
                elif cell_type == 'Pyramidal Cell':
                    colour_map.append('yellow')

            except KeyError as e:
                # If the neuron exists but does not pass the threshold
                colour_map.append('grey')
        plt.figure()
        plt.title(self.region)
        nx.draw(g, with_labels=True, node_color=colour_map)
        plt.show()




