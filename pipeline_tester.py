from data_loader import *
from curated_data_loader import *
import pandas as pd

if __name__ == "__main__":

    region = "VISam"
    counts_from_raw = DataLoader().retrieve_spike_time_matrix_per_region(region)
    counts_from_curated = CuratedDataLoader.load_data(region)

    counts_equal = counts_from_raw.equals(counts_from_curated)
    i = 0