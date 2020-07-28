import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

import numpy as np
from scipy import stats
from itertools import count
import math


class SubnetworkFinder:
    """
    Class to detect and visualise functional networks in specific regions

    Methods:

    """
    @staticmethod
    def above_percentile(cell_value, flattened_matrix, percentile):
        if math.isnan(cell_value):
            return False
        return stats.percentileofscore(flattened_matrix, cell_value) >= percentile

    @staticmethod
    def find_functional_subnetwork(adjacency_matrix, percentile=75):
        mask = np.triu(np.ones_like(adjacency_matrix, dtype=np.bool))
        upper_half = adjacency_matrix.where(mask).dropna(axis='columns', how='all').dropna(axis='rows', how='all')

        flattened_matrix = upper_half.to_numpy().flatten()
        flattened_matrix = flattened_matrix[~np.isnan(flattened_matrix)]

        transformed_upper_half = upper_half.applymap(
            lambda cell: SubnetworkFinder.above_percentile(cell, flattened_matrix, percentile))

        np.fill_diagonal(transformed_upper_half.values, False)

        # transformed_upper_half now has true when there is an 'above-threshold' connection
        # between neurons

        return transformed_upper_half.columns[(transformed_upper_half == True).any(axis=1)], transformed_upper_half

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

    def create_histogram_of_cell_types(self, cells_with_extra_info):
        sns.catplot(x="Cell_type", kind="count", palette="ch:.25", data=cells_with_extra_info)  #.set_title(self.region)
        plt.show()

    # TODO: want to finish this off so that we colour code the nodes by cell type
    # which means figuring out how networkx works
    def create_graph_diagram(self, thresholded_adjacency_matrix, cells_with_type):

        # Create the graph without isolates
        g = nx.from_pandas_adjacency(thresholded_adjacency_matrix)
        g.remove_nodes_from(list(nx.isolates(g)))

        # Find all of the cell types for colour scheme
        cell_groups = cells_with_type['Cell_type'].unique()
        mapping = dict(zip(sorted(cell_groups), count()))

        nodes = g.nodes()
        #colours = [mapping[g.nodes[n].n] for n in nodes]

        color_lookup = {k: v for v, k in enumerate(sorted(set(g.nodes())))}
        low, *_, high = sorted(color_lookup.values())
        norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

        nx.draw(g, with_labels=True, nodelist=color_lookup, node_color=[mapper.to_rgba(i) for i in color_lookup.values()])
        plt.show()

        """
        import matplotlib.pyplot as plt
        # create number for each group to allow use of colormap
        from itertools import count
        # get unique groups
        groups = set(nx.get_node_attributes(g,'group').values())
        mapping = dict(zip(sorted(groups),count()))
        
        nodes = g.nodes()
        colors = [mapping[g.node[n]['group']] for n in nodes]
        
        # drawing nodes and edges separately so we can capture collection for colobar
        pos = nx.spring_layout(g)
        ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, 
                                    with_labels=False, node_size=100, cmap=plt.cm.jet)
        plt.colorbar(nc)
        plt.axis('off')
        plt.show()"""


