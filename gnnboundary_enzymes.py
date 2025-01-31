from gnnboundary import *
from gnn_xai_common import *
import numpy as np
from tqdm.auto import tqdm, trange
import torch
from torch import nn
import torch_geometric as pyg
from torchmetrics import F1Score, ConfusionMatrix, Accuracy
import random
import matplotlib.pyplot as plt
import seaborn as sns 
from gnninterpreter_enzymes import *
import os 


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main_enzymes():
    
    #set seed
    seed = 3407

    # Create 'logs' directory (if it doesn't exist)
    logs_directory = "logs"
    figures_directory = "figures"
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
    if not os.path.exists(figures_directory):
        os.makedirs(figures_directory)

    enzymes_log(seed)


def enzymes_log(seed):
    with open("logs/convergence_enzymes.txt", "w") as log_file_convergence, open("logs/analysis_enzymes.txt", "w") as log_file_analysis, open("logs/Interpreter_enzymes.txt", "w") as log_file_interpreter_probs:
        # Write header
        log_file_convergence.write("Seed\tClassPair\tSuccessRate\tAvgConvergenceIteration\n")
        log_file_analysis.write("Seed\tClassPair\tMeanClassProb\tStd\tMargin1\tMargin2\tThickness1\tThickness2\tComplexity\n")
        log_file_interpreter_probs.write("Seed\tClass\tMeanProbability\tStd\n")

        # Initialize dataset
        enzymes = ENZYMESDataset(seed=12345)

        # Seed all
        seed_all(seed) 

        # Load model
        model = GCNClassifier(node_features=len(enzymes.NODE_CLS),
                            num_classes=len(enzymes.GRAPH_CLS),
                            hidden_channels=32,
                            num_layers=3)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.load_state_dict(torch.load('ckpts/enzymes.pt', map_location=torch.device(device)))
        
        dataset_list_gt = enzymes.split_by_class()
        dataset_list_pred = enzymes.split_by_pred(model)

        model.to(device)
        evaluation = enzymes.model_evaluate(model)
        model.to('cpu')
        evaluation

        # Get Mean Embeddings 
        embeds = [[] for _ in range(len(enzymes.GRAPH_CLS))]
        with torch.no_grad():
            for data in tqdm(enzymes):
                embeds[data.y.item()].append(model.eval()(pyg.data.Batch.from_data_list([data]))["embeds"].numpy())
        mean_embeds = [torch.tensor(np.concatenate(e).mean(axis=0)) for e in embeds]

        print("Generating Confusion Matrix and Adjacency Matrix")
        save_matrix(evaluation['cm'], enzymes.GRAPH_CLS.values(), xlabel = 'Predicted', ylabel = 'Actual', fmt='d', filename=f"figures/confusion_matrix_enzymes_{seed}.png")
        adj_ratio_mat, boundary_info = pairwise_boundary_analysis(model, dataset_list_pred)
        save_matrix(adj_ratio_mat, enzymes.GRAPH_CLS.values(), xlabel = None, ylabel = None, fmt='.2f', filename=f"figures/adjacency_matrix_enzymes_{seed}.png")

        # Generate class grpahs with GNNInterpreter for boundary analysis 
        print("Generating class graphs with GNNInterpreter for boundary analysis")

        GNNInterpreterSamples = GNNInterpreter_TrainandSampleGraphs(enzymes, seed, log_file_interpreter_probs)


        print("Main Analysis")
        # Training and Case Study for each class pair (0,3), (0,4), (0,5), (1,2), (3,4), (4,5)
        trainer = {(0,3): None, (0,4): None, (0,5): None, (1,2): None, (3,4): None, (4,5): None}
        boundary_margin = {(0,3): None, (3,0): None, (0,4): None, (4,0): None, (0,5): None, (5,0): None, 
                            (1,2): None, (2,1): None, (3,4): None, (4,3): None, (4,5): None, (5,4): None}
        boundary_thickness = {(0,3): None, (3,0): None, (0,4): None, (4,0): None, (0,5): None, (5,0): None,
                            (1,2): None, (2,1): None, (3,4): None, (4,3): None, (4,5): None, (5,4): None}

        # Class pair: 0, 3
        print("Running class 0 and 3")
        trainer[(0,3)] = class0and3(enzymes, mean_embeds, model, trainer, seed, log_file_convergence) 
        boundary_margin[(0,3)], boundary_margin[(3,0)], boundary_thickness[(0,3)], boundary_thickness[(3,0)] = trainer[(0,3)].quantatitive(GNNInterpreterSamples, seed, (0,3), log_file_analysis)

        seed_all(seed)

        # Class pair: 0, 4
        print("Running class 0 and 4")
        trainer[(0,4)] = class0and4(enzymes, mean_embeds, model, trainer, seed, log_file_convergence)
        boundary_margin[(0,4)], boundary_margin[(4,0)], boundary_thickness[(0,4)], boundary_thickness[(4,0)] = trainer[(0,4)].quantatitive(GNNInterpreterSamples, seed, (0,4), log_file_analysis)

        seed_all(seed)

        # Class pair: 0, 5
        print("Running class 0 and 5")
        trainer[(0,5)] = class0and5(enzymes, mean_embeds, model, trainer, seed, log_file_convergence)
        boundary_margin[(0,5)], boundary_margin[(5,0)], boundary_thickness[(0,5)], boundary_thickness[(5,0)] = trainer[(0,5)].quantatitive(GNNInterpreterSamples, seed, (0,5), log_file_analysis)

        seed_all(seed)

        # Class pair: 1, 2 
        print("Running class 1 and 2")
        trainer[(1,2)] = class1and2(enzymes, mean_embeds, model, trainer, seed, log_file_convergence)
        boundary_margin[(1,2)], boundary_margin[(2,1)], boundary_thickness[(1,2)], boundary_thickness[(2,1)] = trainer[(1,2)].quantatitive(GNNInterpreterSamples, seed, (1,2), log_file_analysis)

        seed_all(seed)

        # Class pair: 3, 4
        print("Running class 3 and 4")
        trainer[(3,4)] = class3and4(enzymes, mean_embeds, model, trainer, seed, log_file_convergence)
        boundary_margin[(3,4)], boundary_margin[(4,3)], boundary_thickness[(3,4)], boundary_thickness[(4,3)] = trainer[(3,4)].quantatitive(GNNInterpreterSamples, seed, (3,4), log_file_analysis)

        seed_all(seed)

        # Class pair: 4, 5
        print("Running class 4 and 5")
        trainer[(4,5)] = class4and5(enzymes, mean_embeds, model, trainer, seed, log_file_convergence)
        boundary_margin[(4,5)], boundary_margin[(5,4)], boundary_thickness[(4,5)], boundary_thickness[(5,4)] = trainer[(4,5)].quantatitive(GNNInterpreterSamples, seed, (4,5), log_file_analysis)

        # Reproduce matrices in Figure 3 (except for confusion matrix which was reproduced above)
        margin_matrix = np.zeros((7, 7))
        thickness_matrix = np.zeros((7, 7))

        # Fill matrices based on dictionary values
        for (i, j), value in boundary_margin.items():
            margin_matrix[i][j] = value  # i=0 means first row, j=0 means first column

        for (i, j), value in boundary_thickness.items():
            thickness_matrix[i][j] = value  # same here

        # Save the matrices
        save_matrix(margin_matrix, enzymes.GRAPH_CLS.values(), boundary=True, xlabel = 'Decision Boundary', ylabel = 'Decision Region', fmt='.2f', filename=f"figures/margin_matrix_enzymes_{seed}.png")
        save_matrix(thickness_matrix, enzymes.GRAPH_CLS.values(), boundary=True, xlabel = 'Decision Boundary', ylabel = 'Decision Region', fmt='.2f', filename=f"figures/thickness_matrix_enzymes_{seed}.png")


def class0and3(enzymes, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 0, 3

    succ_list = []
    iteration_list = []

    best_trainer = None

    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=35,
                temperature=0.15,
                num_node_cls=len(enzymes.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=8000),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=5),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=5),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=enzymes,
            seed = seed, # Added
            classes = (cls_1, cls_2), # Added
            budget_penalty=BudgetPenalty(budget=24, order=2, beta=0.8),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.4, 0.6), cls_2: (0.4, 0.6)},
            target_size=250,
            w_budget_init=1.4,
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

    log_file_convergence.write(f"{seed}\t{(0,3)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(cls_1, cls_2)] if best_trainer is None else best_trainer

def class0and4(enzymes, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 0, 4

    succ_list = []
    iteration_list = []

    best_trainer = None

    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=35,
                temperature=0.15,
                num_node_cls=len(enzymes.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=5000),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=10),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=10),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=3),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=enzymes,
            seed = seed, # Added
            classes = (cls_1, cls_2), # Added
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=0.8),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.4, 0.6), cls_2: (0.4, 0.6)},
            target_size=200,
            w_budget_init=1.4,
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

    log_file_convergence.write(f"{seed}\t{(0,4)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(cls_1, cls_2)] if best_trainer is None else best_trainer



def class0and5(enzymes, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 0, 5

    succ_list = []
    iteration_list = []

    best_trainer = None

    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=35,
                temperature=0.15,
                num_node_cls=len(enzymes.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=5000),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=20),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=20),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=2),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=enzymes,
            seed = seed, # Added
            classes = (cls_1, cls_2), # Added
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1.8),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.4, 0.6), cls_2: (0.4, 0.6)},
            target_size=250,
            w_budget_init=1.4,
            w_budget_inc=1.2,
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

    log_file_convergence.write(f"{seed}\t{(0,5)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(cls_1, cls_2)] if best_trainer is None else best_trainer


def class1and2(enzymes, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 1, 2

    succ_list = []
    iteration_list = []

    best_trainer = None

    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=35,
                temperature=0.15,
                num_node_cls=len(enzymes.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=30),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=5),
                dict(key="omega", criterion=NormPenalty(order=2), weight=2),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=enzymes,
            seed = seed, # Added
            classes = (cls_1, cls_2), # Added
            budget_penalty=BudgetPenalty(budget=24, order=2, beta=0.6),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.4, 0.6), cls_2: (0.4, 0.6)},
            target_size=150,
            w_budget_init=1.4,
            w_budget_inc=1.2,
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

    log_file_convergence.write(f"{seed}\t{(1,2)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(cls_1, cls_2)] if best_trainer is None else best_trainer


def class3and4(enzymes, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 3, 4

    succ_list = []
    iteration_list = []

    best_trainer = None

    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=35,
                temperature=0.15,
                num_node_cls=len(enzymes.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=30),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=5),
                dict(key="omega", criterion=NormPenalty(order=2), weight=2),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=enzymes,
            seed = seed, # Added
            classes = (cls_1, cls_2), # Added
            budget_penalty=BudgetPenalty(budget=24, order=2, beta=0.6),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.4, 0.6), cls_2: (0.4, 0.6)},
            target_size=150,
            w_budget_init=1.4,
            w_budget_inc=1.2,
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

    log_file_convergence.write(f"{seed}\t{(3,4)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(cls_1, cls_2)] if best_trainer is None else best_trainer


def class4and5(enzymes, mean_embeds, model, trainer, seed, log_file_convergence):
    cls_1, cls_2 = 4, 5

    succ_list = []
    iteration_list = []

    best_trainer = None

    for i in range(1000):
        print("Iteration: ", i)
        trainer[(cls_1, cls_2)] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=35,
                temperature=0.15,
                num_node_cls=len(enzymes.NODE_CLS),
                learn_node_feat=True
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=DynamicBalancingBoundaryCriterion(classes=[cls_1, cls_2]), weight=30),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_1]), weight=0),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[cls_2]), weight=0),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=5),
                dict(key="omega", criterion=NormPenalty(order=2), weight=2),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=enzymes,
            seed = seed, # Added
            classes = (cls_1, cls_2), # Added
            budget_penalty=BudgetPenalty(budget=24, order=2, beta=0.6),
        )

        succ, iteration = trainer[(cls_1, cls_2)].train(
            iterations=500,
            target_probs={cls_1: (0.45, 0.55), cls_2: (0.45, 0.55)},
            target_size=150,
            w_budget_init=1.4,
            w_budget_inc=1.2,
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

    log_file_convergence.write(f"{seed}\t{(4,5)}\t{np.mean(succ_list)}\t{iteration_list}\n")

    return trainer[(cls_1, cls_2)] if best_trainer is None else best_trainer



