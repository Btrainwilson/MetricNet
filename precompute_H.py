import numpy as np
import pyexml as pyml

domain_set = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_n314_m16.npy")

H_dataset = pyml.datasets.MetricSpaceDataset(pyml.geometry.metrics.numpy_metrics.IndependentSet(), domain_set, precompute=True)
H = H_dataset.getPreCompute()

np.save(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_PairwiseMetric_HammingLattice_n314_m16.npy", H)