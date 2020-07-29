from curated_data_loader import *
from subnetwork_finder import *

import networkx as nx


class SubnetworkAnalysis:

    @staticmethod
    def identify_subnetwork_across_sessions(regions):
        """
        Extract the functional subnetwork for the regions across sessions
        Analyse the putative subnetwork (e.g. using graph features)
        Compare the networks across sessions
        :param regions:
        :return:
        """

        downloaded_data = CuratedDataLoader.download_data()
        mouse_name = "Lederberg"
        session_indexes = CuratedDataLoader.indexes_of_sessions_for_mouse(mouse_name, downloaded_data)

        # Proportion_Pyr = number of cells that are pyramidal cells in the identified network
        # Proportion_Wide_IN etc
        output = pd.DataFrame(columns=['Session',
                                       'Region',
                                       'Cell_IDs',
                                       'Prop_Cells',
                                       'Prop_Pyr',
                                       'Prop_Wide_In',
                                       'Prop_Narrow_IN',
                                       'Mean_Degree',
                                       'Clustering_Coeff'])
        threshold = 90
        for session in session_indexes:
            for region in regions:
                extracted_data = CuratedDataLoader(session).spikes_in_decision_time_per_neuron(region, downloaded_data)
                finder = SubnetworkFinder()
                adjacency_matrix = finder.find_network_by_linear_correlation_full_window(extracted_data.activity_matrix)
                shuffled_adjacency_matrix = finder.find_adjacency_matrix_from_shuffled_data(extracted_data.activity_matrix)
                adjacency_matrix -= shuffled_adjacency_matrix

                cells, cell_network = finder.find_functional_subnetwork(adjacency_matrix, threshold)
                g = finder.create_graph_of_network(cell_network, threshold)

                proportion_cells = SubnetworkAnalysis.proportion_cells_in_subnetwork(adjacency_matrix, cells)
                cell_type_props = SubnetworkAnalysis.proportion_cell_types_in_subnetwork(cells, extracted_data.session)

                mean_degree = sum(dict(g.degree()).values())/float(len(g)) if len(g) > 0 else 0
                clustering_coeff = nx.average_clustering(g) if len(g) > 0 else 0
                new_row = pd.Series([
                        extracted_data.session.mouse_name + "_" + extracted_data.session.session_date,
                        region,
                        np.array_str(cells),
                        proportion_cells,
                        SubnetworkAnalysis.proportion_cell_type(cell_type_props, 'Pyramidal Cell'),
                        SubnetworkAnalysis.proportion_cell_type(cell_type_props, 'Wide Interneuron'),
                        SubnetworkAnalysis.proportion_cell_type(cell_type_props, 'Narrow Interneuron'),
                        mean_degree,
                        clustering_coeff
                    ], index=output.columns)
                output = output.append(new_row, ignore_index=True)

        output.to_csv(mouse_name + "_all_sessions")

    @staticmethod
    def identify_network_and_visualise(region):
        extracted_data = CuratedDataLoader(11).spikes_in_decision_time_per_neuron(region)

        finder = SubnetworkFinder()
        adjacency_matrix = finder.find_network_by_linear_correlation_full_window(extracted_data.activity_matrix)

        shuffled_adjacency_matrix = finder.find_adjacency_matrix_from_shuffled_data(extracted_data.activity_matrix)

        adjacency_matrix -= shuffled_adjacency_matrix
        visualiser = SubnetworkVisualiser(region)

        visualiser.create_histogram_of_correlations(adjacency_matrix, 95)
        visualiser.create_heatmap_from_adjacency_matrix(adjacency_matrix)

        threshold = 95
        print("Percentile Threshold " + str(threshold))
        cells, cell_network = finder.find_functional_subnetwork(adjacency_matrix, threshold)

        count_in_network = len(cells)
        print("Number of cells in subnetwork " + str(count_in_network))
        all_cell_count = len(adjacency_matrix)
        print("Number of cells in original recording from that region " + str(all_cell_count))
        print("% of measured cells identified as being in the network " + str(count_in_network/all_cell_count*100))
        cells_with_type = CuratedDataLoader.neuron_id_to_cell_type(cells,
                                                                    "data/CellMeasures_" +
                                                                    extracted_data.session.mouse_name + "_" +
                                                                    extracted_data.session.session_date +
                                                                    ".csv")
        visualiser.create_histogram_of_cell_types(cells_with_type)

        visualiser.create_graph_diagram(cell_network, cells_with_type, False)

    @staticmethod
    def proportion_cells_in_subnetwork(adjacency_matrix, cells):
        measured_neurons = len(adjacency_matrix)
        if measured_neurons < 1:
            return 0
        return len(cells)/measured_neurons

    @staticmethod
    def proportion_cell_types_in_subnetwork(cells, session_info):
        cells_with_type = CuratedDataLoader.neuron_id_to_cell_type(cells,
                                                                   "data/CellMeasures_" +
                                                                   session_info.mouse_name + "_" +
                                                                   session_info.session_date +
                                                                   ".csv")
        counts_of_cell_types = cells_with_type['Cell_type'].value_counts()
        return counts_of_cell_types/len(cells)

    @staticmethod
    def proportion_cell_type(proportion_map, cell_type):
        try:
            return proportion_map[cell_type]
        except KeyError as e:
            return 0


if __name__ == "__main__":
    #SubnetworkAnalysis.identify_network_and_visualise("VISam")
    SubnetworkAnalysis.identify_subnetwork_across_sessions(["VISam", "ACA", "MOs", "PL", "MOp", "SSp", "APN"])


