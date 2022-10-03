import pyexlab as pylab
import pyexml as pyml
import torch
import os

import numpy as np

#Data sets
domain_set = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\IS_n314_m16.npy")
image_set = np.load(r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\datasets\Lattice 4x4 Independent Set\Lattice_n314_m16.npy")

#Construct datasets
map_dataset = pyml.datasets.MapDataset(domain_set / 1000, image_set / 1000)

#Construct neural networks
map_net = pyml.models.Simple_FFWDNet(domain_set.shape[1], image_set.shape[1])

#Construct trainers
map_optimizer = torch.optim.SGD(map_net.parameters(), lr=0.0001)
map_trainer = pyml.trainers.DynamicLSATrainer(map_net, map_dataset, torch.nn.MSELoss(), optimizer = map_optimizer, scheduler = torch.optim.lr_scheduler.ExponentialLR(map_optimizer, gamma=0.999))
map_tester = pyml.trainers.Tester(map_net, map_dataset, torch.nn.MSELoss())

#Construct the coach that invokes the trainers at certain epochs
trainer_schedule = pyml.trainers.Modulus_Schedule([-1, -1])
trainers = [map_trainer, map_tester]

map_coach = pyml.trainers.Coach(trainers=trainers, trainer_schedule=trainer_schedule)
map_subject = pyml.mlsubjects.NeuralNetSubject(map_coach)

#Build experiment
folder_save_path = r"C:\Users\bwilson\OneDrive - QuEra Computing\Documents\research_code\ISMetric\experiments\EX01_Dynamic_Metric\trials\map_trials"
experiment_name = "experiment_" + pylab.fio.datetime_now_str()
experiment_save_path = pylab.fio.makeDirectory(folder_save_path, experiment_name)
experiment = pylab.Experiment(experiment_save_path, [map_subject])

#Run experiment
experiment.run(epochs=100) #Check formatting of output of model to be batch conscious and type appropriate