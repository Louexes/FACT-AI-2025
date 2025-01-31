from gnnboundary import *
from gnn_xai_common import *
from gnninterpreter_collab import *
import numpy as np
from tqdm.auto import tqdm, trange
import torch
from torch import nn
import torch_geometric as pyg
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
import random
import os 


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main_collab():

    #set seed
    seed = 3407

    # Create 'logs' directory (if it doesn't exist)
    logs_directory = "logs"
    figures_directory = "figures"
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    collab_log(seed)


def collab_log(seed):
    with open("logs/convergence_collab.txt", "w") as log_file_convergence, open("logs/analysis_collab.txt", "w") as log_file_analysis, open("logs/Interpreter_collab.txt", "w") as log_file_interpreter_probs:
        # Write header
        log_file_convergence.write("Seed\tClassPair\tSuccessRate\tAvgConvergenceIteration\n")
        log_file_analysis.write("Seed\tClassPair\tMeanClassProb\tStd\tMargin1\tMargin2\tThickness1\tThickness2\tComplexity\n")
        log_file_interpreter_probs.write("Seed\tClass\tMeanProbability\tStd\n")

        # Initialize dataset
        collab = CollabDataset(seed=12345)

        # Seed all
        seed_all(seed) 

        # Load model
        model = GCNClassifier(node_features=len(collab.NODE_CLS),
                            num_classes=len(collab.GRAPH_CLS),
                            hidden_channels=64,
                            num_layers=5)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.load_state_dict(torch.load('ckpts/collab.pt', map_location=torch.device(device)))
        
        dataset_list_gt = collab.split_by_class()
        dataset_list_pred = collab.split_by_pred(model)

        model.to(device)
        evaluation = collab.model_evaluate(model)
        model.to('cpu')
        evaluation

        # Get Mean Embeddings
        embeds = [[] for _ in range(len(collab.GRAPH_CLS))]
        with torch.no_grad():
            for data in tqdm(collab):
                embeds[data.y.item()].append(model.eval()(pyg.data.Batch.from_data_list([data]))["embeds"].numpy())
        mean_embeds = [torch.tensor(np.concatenate(e).mean(axis=0)) for e in embeds]

        print("Generating Confusion Matrix and Adjacency Matrix")
        save_matrix(evaluation['cm'], collab.GRAPH_CLS.values(), xlabel = 'Predicted', ylabel = 'Actual', fmt='d', filename=f"figures/confusion_matrix_collab_{seed}.png")
        adj_ratio_mat, boundary_info = pairwise_boundary_analysis(model, dataset_list_pred)
        save_matrix(adj_ratio_mat, collab.GRAPH_CLS.values(), xlabel=None, ylabel=None, fmt='.2f', filename=f"figures/adjacency_matrix_collab_{seed}.png")

        print("Generating class graphs with GNNInterpreter for boundary analysis")

        GNNInterpreterSamples = GNNInterpreter_TrainandSampleGraphs(collab, seed, log_file_interpreter_probs) #Train GNNInterpreter and sample graphs from each class

        print("Main Analysis")
        # Training and Case Study for each class pair (0,1) and (0,2)
        trainer = {(0,1): None, (0,2): None}
        boundary_margin = {(0,1): None, (1,0): None, (0,2): None, (2,0): None}
        boundary_thickness = {(0,1): None, (1,0): None, (0,2): None, (2,0): None}

        # Class pair: 0, 1
        print("Running class 0 and 1")
        trainer[(0,1)] = class0and1(collab, mean_embeds, model, trainer, seed, log_file_convergence)
        boundary_margin[(0,1)], boundary_margin[(1,0)], boundary_thickness[(0,1)], boundary_thickness[(1,0)] = trainer[(0,1)].quantatitive(GNNInterpreterSamples, seed, (0,1), log_file_analysis)

        seed_all(seed)

        # Class pair: 0, 2
        print("Running class 0 and 2")
        trainer[(0,2)] = class0and2(collab, mean_embeds, model, trainer, seed, log_file_convergence) #Returns trainer model
        boundary_margin[(0,2)], boundary_margin[(2,0)], boundary_thickness[(0,2)], boundary_thickness[(2,0)] = trainer[(0,2)].quantatitive(GNNInterpreterSamples, seed, (0,2), log_file_analysis) #Gets mean probabilities of 500 graphs and computes

        # Reproduce matrices in Figure 3 (except for confusion matrix which was reproduced above)
        margin_matrix = np.zeros((3, 3))
        thickness_matrix = np.zeros((3, 3))

        # Fill matrices based on dictionary values
        for (i, j), value in boundary_margin.items():
            margin_matrix[i][j] = value  # i=0 means first row, j=0 means first column

        for (i, j), value in boundary_thickness.items():
            thickness_matrix[i][j] = value  # same here

        # Save the matrices
        save_matrix(margin_matrix, collab.GRAPH_CLS.values(), boundary=True, xlabel = 'Decision Boundary', ylabel = 'Decision Region', fmt='.2f', filename=f"figures/margin_matrix_collab_{seed}.png")
        save_matrix(thickness_matrix, collab.GRAPH_CLS.values(), boundary=True, xlabel = 'Decision Boundary', ylabel = 'Decision Region', fmt='.2f', filename=f"figures/thickness_matrix_collab_{seed}.png")



def class0and1(collab, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 0, 1

    succ_list = []
    iteration_list = []

    best_trainer = None

    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=25,
                temperature=0.15,
                num_node_cls=len(collab.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=25),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=collab,
            seed = seed, # Added
            classes = (cls_1, cls_2), # Added
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.45, 0.55), cls_2: (0.45, 0.55)},
            target_size=60,
            w_budget_init=1,
            w_budget_inc=1.3,
            w_budget_dec=0.8,
            k_samples=32,
        )

        succ_list.append(succ)
        
        if succ:
            iteration_list.append(iteration)
            if best_trainer is None: #In case the last iteration does not succeed, we always return trainer from the first successful iteration
                best_trainer = trainer[(cls_1, cls_2)]
            else:
                trainer[(cls_1, cls_2)] = best_trainer

    succ_list = np.array(succ_list)
    iteration_list = np.array(iteration_list)

    if iteration_list.size > 0:
        iteration_list = np.mean(iteration_list)
    else:
        iteration_list = "No Success"

    log_file_convergence.write(f"{seed}\t{(0,1)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(0, 1)] if best_trainer is None else best_trainer



def class0and2(collab, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 0, 2

    succ_list = []
    iteration_list = []

    best_trainer = None


    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=25,
                temperature=0.15,
                num_node_cls=len(collab.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=25),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=collab,
            seed = seed, 
            classes = (cls_1, cls_2), 
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.45, 0.55), cls_2: (0.45, 0.55)},
            target_size=60,
            w_budget_init=1,
            w_budget_inc=1.3,
            w_budget_dec=0.8,
            k_samples=32,
        )

        succ_list.append(succ)
        
        if succ:
            iteration_list.append(iteration)
            if best_trainer is None: #In case the last iteration does not succeed, we always return trainer from the first successful iteration
                best_trainer = trainer[(cls_1, cls_2)]
            else:
                trainer[(cls_1, cls_2)] = best_trainer
        

    succ_list = np.array(succ_list)
    iteration_list = np.array(iteration_list)

    if iteration_list.size > 0:
        iteration_list = np.mean(iteration_list)
    else:
        iteration_list = "No Success"

    log_file_convergence.write(f"{seed}\t{(0,1)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(0, 2)] if best_trainer is None else best_trainer




