import pyexlab as pylab
import pyexml as pyml
import torch
import os
import copy
import numpy as np
from run_dynamic_full import dynamic_metric_subject

if __name__ == "__main__":

    domain_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_n314_m16.npy"
    image_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\Lattice_n314_m16.npy"
    precompute = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_PairwiseMetric_HammingLattice_n314_m16.npy")

    metric_subject = dynamic_metric_subject(domain_path, image_path, precompute)

    #Set to static map instead of dynamic
    metric_subject_static = copy.deepcopy(metric_subject)
    metric_subject_static.trainer.trainers[1].epoch_mod = -1
    metric_subject_static.__name__ = "Static Map Subject"


    #Build experiment
    folder_save_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\dynamic_static_compare\4x4"
    experiment_name = "experiment_" + pylab.fio.datetime_now_str()
    experiment_save_path = pylab.fio.makeDirectory(folder_save_path, experiment_name)
    experiment = pylab.Experiment(experiment_save_path, [metric_subject, metric_subject_static])

    #Run experiment
    experiment.run(epochs=10) #Check formatting of output of model to be batch conscious and type appropriate