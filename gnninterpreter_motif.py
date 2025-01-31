from gnnboundary import * #gnnboundary has been adjusted to incorporate any files need by gnninterpreter. they are similar to begin with.
from gnn_xai_common import *

import numpy as np
import matplotlib.pyplot as plt

from gnn_xai_common import *

from tqdm.auto import tqdm, trange

import torch
from torch import nn
import torch_geometric as pyg
from torchmetrics import F1Score
import random

import os 

def GNNInterpreter_TrainandSampleGraphs(motif, seed, log_file_interpreter_probs):
    """
    Trains the GNNInterpreter for each class and generates samples from each class.
    
    Args:
        motif (MotifDataset): The collab dataset to be used for training and sampling.
        seed (int): The random seed for reproducibility.
    
    Returns:
        dict: Dictionary containing generated samples for each class, with class indices as keys.
    """

    trainer = {}
    sampleDict = {} # Dictionary to store the samples for each class, keys are cls_idx and values are lists of tensors (embeddings of graphs)

    model = GCNClassifier(node_features=len(motif.NODE_CLS),
                        num_classes=len(motif.GRAPH_CLS),
                        hidden_channels=6,
                        num_layers=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load('ckpts/motif.pt', map_location=torch.device(device)))
    
    dataset_list_gt = motif.split_by_class()
    dataset_list_pred = motif.split_by_pred(model)

    mean_embeds = [d.model_transform(model, key="embeds").mean(dim=0) for d in dataset_list_gt]


    for cls_idx in [0,1,2, 3]:

        optimal_hyperparameters = {0: {'ClassScoreCriterionMaximizeWeight': 50, 'BudgetPenaltyBeta': 0.5},
                                1: {'ClassScoreCriterionMaximizeWeight': 50, 'BudgetPenaltyBeta': 0.5},
                                    2: {'ClassScoreCriterionMaximizeWeight': 50, 'BudgetPenaltyBeta': 0.5},
                                    3: {'ClassScoreCriterionMaximizeWeight': 100, 'BudgetPenaltyBeta': 0.8}}

        trainer[cls_idx] = Trainer(
        sampler=(s := GraphSampler(
            max_nodes=30,
            num_node_cls=len(motif.NODE_CLS),
            temperature=0.15,
            learn_node_feat=True
        )),
        discriminator=model,
        criterion=WeightedCriterion([
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=cls_idx, mode='maximize'), weight=optimal_hyperparameters[cls_idx]['ClassScoreCriterionMaximizeWeight']),
            dict(key="logits", criterion=ClassScoreCriterion(class_idx=1, mode='minimize'), weight=0),
            dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_idx]), weight=5),
            dict(key="logits", criterion=MeanPenalty(), weight=0),
            dict(key="omega", criterion=NormPenalty(order=1), weight=1),
            dict(key="omega", criterion=NormPenalty(order=2), weight=1),
            dict(key="xi", criterion=NormPenalty(order=1), weight=0),
            dict(key="xi", criterion=NormPenalty(order=2), weight=0),
            # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
            # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
            dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=1),
        ]),
        optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
        scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
        dataset=motif,
        seed = seed,
        classes= cls_idx,
        budget_penalty=BudgetPenalty(budget=10, order=2, beta=optimal_hyperparameters[cls_idx]['BudgetPenaltyBeta'])
        )

        convergence = False
        counter = 0

        while convergence == False and counter <= 100: #Note: Convergence may not occur the first time, so we allow for up to 100 iterations. Else load in checkpoints.
        
            print(f"Training GNNInterpreter for class {cls_idx}")

            convergence, _ = trainer[cls_idx].train(
            iterations=2000,
            target_probs={cls_idx: (0.9, 1.0)},
            target_size=120,
            w_budget_init=1,
            w_budget_inc=1.3,
            w_budget_dec=0.80,
            k_samples=16
            )
            
            if convergence == False:
                print(f"Convergence did not occur, retrying.... Attempt: {counter + 1} out of 100")
                counter += 1

        if convergence == True:
            print(f"Converged for class {cls_idx}")
            torch.save(trainer[cls_idx].sampler.state_dict(), f'Interpreter-ckpts/motif{cls_idx}.pt')
        else:
            print(f"Did not converge for class {cls_idx}. Loading provided checkpoint...")
            trainer[cls_idx].sampler.load_state_dict(torch.load(f'Interpreter-ckpts/motif{cls_idx}.pt'))
    

    print(f"Generating samples with GNNInterpreter for each class")

    for cls_idx in [0,1,2, 3]:
        samples, mean_probs, std_probs = trainer[cls_idx].quantatitiveInterpreter() # We log the mean of the generated samples to ensure they are faithful to the class (i.e mean probability > 0.9)
        sampleDict[cls_idx] = samples
        log_file_interpreter_probs.write(f"{seed}\t{cls_idx}\t{mean_probs}\t{std_probs}\n")
    
    return sampleDict




    


