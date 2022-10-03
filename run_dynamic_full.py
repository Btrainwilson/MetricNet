import pyexlab as pylab
import pyexml as pyml
import torch
import os
import copy
import numpy as np

def dynamic_metric_subject(domain_set_path, image_set_path, metric_precompute):

    #Data sets
    domain_set = np.load(domain_set_path)
    image_set = np.load(image_set_path)

    domain_set = domain_set / 1000
    image_set = image_set / 1000

    n = len(domain_set)
    train_test_ratio = 0.7
    subspace_ratio = 0.5

    total_datasize = int(n * subspace_ratio)
    cutoff_idx = int(total_datasize * train_test_ratio)

    #Split Indeces for datasets
    subset_idx = pyml.datasets.utils.split_indeces(n, cutoff_idx, total_datasize)

    #Map datasets
    map_training_dataset = pyml.datasets.MapDataset(domain_set, image_set, subspace=subset_idx[0])
    map_testing_dataset = pyml.datasets.MapDataset(domain_set, image_set, subspace=subset_idx[1])

    #Metric datasets
    precompute=metric_precompute

    metric_training_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute=precompute, subspace=subset_idx[0])
    metric_testing_dataset = pyml.datasets.MetricSpaceDataset(metric = None, space = domain_set, precompute=precompute, subspace=subset_idx[1])

    #Construct neural networks
    map_net = pyml.models.Simple_FFWDNet(domain_set.shape[1], image_set.shape[1])
    metric_net = pyml.models.MetricNet(map_net, pyml.geometry.metrics.torch_metrics.Euclidean())

    #Construct trainers
    metric_optimizer = torch.optim.Adam(metric_net.parameters(), lr=0.0001)
    map_optimizer = torch.optim.Adam(map_net.parameters(), lr=0.0001)

    map_trainer = pyml.trainers.DynamicLSATrainer(map_net, map_training_dataset, torch.nn.MSELoss(), 
                                                    optimizer = map_optimizer, 
                                                    scheduler = torch.optim.lr_scheduler.ExponentialLR(metric_optimizer, gamma=0.99),
                                                    alt_name="Map Trainer", epoch_mod=10)
    map_tester  = pyml.trainers.Tester(map_net, map_testing_dataset, torch.nn.MSELoss(), alt_name="Map Tester")

    metric_trainer = pyml.trainers.Trainer(metric_net, metric_training_dataset, torch.nn.MSELoss(), metric_optimizer, 
                                            scheduler = torch.optim.lr_scheduler.ExponentialLR(metric_optimizer, gamma=0.99),
                                            alt_name= "Metric Trainer")
    metric_tester  = pyml.trainers.Tester(metric_net, metric_testing_dataset, torch.nn.MSELoss(), alt_name= "Metric Tester")

    #Construct the coach that invokes the trainers at certain epochs
    trainer_schedule = pyml.trainers.Modulus_Schedule([-1, -1, -1, -1])
    trainers = [metric_trainer, map_trainer, metric_tester, map_tester]

    metric_coach = pyml.trainers.Coach(trainers=trainers, trainer_schedule=trainer_schedule)
    metric_subject = pyml.mlsubjects.NeuralNetSubject(metric_coach, alt_name="Dynamic Map Subject")

    return metric_subject

if __name__ == "__main__":
    domain_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_n314_m16.npy"
    image_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\Lattice_n314_m16.npy"
    precompute = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_PairwiseMetric_HammingLattice_n314_m16.npy")

    metric_subject = dynamic_metric_subject(domain_path, image_path, precompute)

    #Build experiment
    folder_save_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\dynamic_full"
    experiment_name = "experiment_" + pylab.fio.datetime_now_str()
    experiment_save_path = pylab.fio.makeDirectory(folder_save_path, experiment_name)
    experiment = pylab.Experiment(experiment_save_path, [metric_subject])

    #Run experiment
    experiment.run(epochs=10) #Check formatting of output of model to be batch conscious and type appropriate