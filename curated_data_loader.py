import numpy as np
import os
import pandas as pd
import requests


class CuratedDataLoader:
    """
    Class to load the data in the same way as is done in
    https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/load_steinmetz_decisions.ipynb
    """
    session_number = 11

    @staticmethod
    def load_data(region):

        fname = CuratedDataLoader.download_data()
        alldat = np.array([])
        for j in range(len(fname)):
            alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz' % j, allow_pickle=True)['dat']))

        dat = alldat[CuratedDataLoader.session_number]

        neurons_with_brain_areas = dat['brain_area']
        indexes_for_neurons_in_region = np.where(neurons_with_brain_areas == region)[0]

        spike_data = dat['spks']
        spikes_for_neurons_in_region = spike_data[indexes_for_neurons_in_region]
        spike_counts_per_neuron = np.count_nonzero(spikes_for_neurons_in_region, axis=2)
        spike_counts_per_neuron_across_all_trials = np.sum(spike_counts_per_neuron, axis=1)

        counts = pd.Series(spike_counts_per_neuron_across_all_trials, index=indexes_for_neurons_in_region)
        return counts

    @staticmethod
    def download_data():
        fname = []
        for j in range(3):
            fname.append('steinmetz_part%d.npz' % j)
        url = ["https://osf.io/agvxh/download"]
        url.append("https://osf.io/uv3mw/download")
        url.append("https://osf.io/ehmw2/download")

        for j in range(len(url)):
            if not os.path.isfile(fname[j]):
                try:
                    r = requests.get(url[j])
                except requests.ConnectionError:
                    print("!!! Failed to download data !!!")
                else:
                    if r.status_code != requests.codes.ok:
                        print("!!! Failed to download data !!!")
                    else:
                        with open(fname[j], "wb") as fid:
                            fid.write(r.content)
        return fname