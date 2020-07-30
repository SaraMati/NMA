from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def visualise_components(component1, component2, labels):
    """
    Plots a 2D representation of the data for visualization with categories
    labelled as different colors.

    Args:
      component1 (numpy array of floats) : Vector of component 1 scores
      component2 (numpy array of floats) : Vector of component 2 scores
      labels (numpy array of floats)     : Vector corresponding to categories of
                                           samples

    Returns:
      Nothing.

    """
    plt.figure()

    color_labels = labels.unique()
    rgb_values = sns.color_palette("Set2", len(color_labels))
    colour_map = dict(zip(color_labels, rgb_values))
    # Translate the colour map into a list of
    colour_scheme = [colour_map[region] for region in labels]
    plt.scatter(x=component1, y=component2, c=colour_scheme)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best')
    plt.show()


def perform_t_sne_on_subnetworks(subnetworks_df):
    subnetworks_df = subnetworks_df.drop(['Session', 'Cell_IDs'], axis=1)
    labels = subnetworks_df['Region']
    subnetworks_df = subnetworks_df.drop(['Region'], axis=1)

    # TODO: are these the right params? copied from tutorial on this
    tsne_model = TSNE(n_components=2, perplexity=20, random_state=2020)
    embed = tsne_model.fit_transform(subnetworks_df)

    # TODO: need to have legend
    visualise_components(embed[:, 0], embed[:, 1], labels)
