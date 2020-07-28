from curated_data_loader import *
from subnetwork_finder import *


def identify_network_and_visualise(region):
    extracted_data = CuratedDataLoader().spikes_in_decision_time_per_neuron(region)

    finder = SubnetworkFinder()
    adjacency_matrix = finder.find_network_by_linear_correlation_full_window(extracted_data.activity_matrix)
    # TODO: other measures of the graph - clustering coefficient? characteristic path length?

    shuffled_adjacency_matrix = finder.find_adjacency_matrix_from_shuffled_data(extracted_data.activity_matrix)

    adjacency_matrix -= shuffled_adjacency_matrix
    visualiser = SubnetworkVisualiser(region)
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

    visualiser.create_graph_diagram(cell_network, cells_with_type)


if __name__ == "__main__":
    identify_network_and_visualise("ACA")
