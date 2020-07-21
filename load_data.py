import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from oneibl.onelight import ONE
one = ONE()

one.set_figshare_url("https://figshare.com/articles/steinmetz/9974357")
sessions = one.search(['spikes'])
session = sessions[11]
one.list(session)

# loads all the objects from the session
spikes = one.load_object(session, 'spikes')
clusters = one.load_object(session, 'clusters')
channels = one.load_object(session, 'channels')
probes = one.load_object(session, 'probes')

# pull out channels from regions of interest (Vis cortex here)
regions = "VIS"
brain_locations = pd.DataFrame(channels.brainLocation)
vis_cx_channels = brain_locations[brain_locations['allen_ontology'].str.contains(regions)]
vis_cx_channels_id = vis_cx_channels.index  #gets the channel ID

# clusters.peakChannel contains the channel number of the location of the peak of the cluster's waveform
# pull out clusters from regions of interest using channel ID
# gets the indices/IDs of clusters from the region of interest
cluster_channels = pd.DataFrame(clusters.peakChannel)
vis_cx_clusters_id = cluster_channels[cluster_channels.isin(vis_cx_channels_id)].dropna().index

# for example, to get waveform durations of all peaks from vis cx clusters:
vis_cx_waveformDuration = clusters.waveformDuration[vis_cx_clusters_id]

# pull out spikes from regions of interest using cluster ID
# gets the spike IDs of spikes from the region of interest
spike_clusters = pd.DataFrame(spikes.clusters)
vis_cx_spikes_id = spike_clusters[spike_clusters.isin(vis_cx_clusters_id)].dropna().index

# to get spike times of spikes from vis cx:
vis_cx_spikes_times = spikes.times[vis_cx_spikes_id]




x_truncated = x[:100].squeeze()

plt.figure()
plt.eventplot(x_truncated)
plt.xlabel("Time (s)")
plt.yticks([])
plt.show

brain_locations = one.load_dataset(session, 'channels.brainLocation')