import torch
import random
import numpy as np


def calculate_boundary_margin(boundary_embeddings_list, class_embeddings_list):
    """
    Calculate the boundary margin for a given class pair.
    
    Args:
        boundary_embeddings_list (list): A list of boundary graph embeddings for a class pair.
        class_embeddings_list (list): A list of graph embeddings for a class.
    
    Returns:
        float: The calculated boundary margin.
    """
    min_distance = float('inf')

    for class_embedding in class_embeddings_list:
        for boundary_embedding in boundary_embeddings_list:
            distance = torch.norm(class_embedding - boundary_embedding, p=2).item()
            if distance < min_distance:
                min_distance = distance

    return min_distance


def boundary_thickness(graph_embeddings_list, boundary_graph_embeddings_list, model_scoring_function, c1, c2, gamma=0.75, num_points=50):
    """
    Args:
        graph_embeddings_list (list): List of torch.Tensor embeddings from graph pooling layer
        boundary_graph_embeddings_list (list): List of torch.Tensor embeddings from graph pooling
                                              layer of boundary graphs
        model_scoring_function: MLP layer after embedding layer
        c1 (int): First class index
        c2 (int): Second class index
        gamma (float): hyperparameter
        num_points (int): number of points used for interpolation

    Returns:
        thickness (float): boundary thickness margin
    """
    
    thickness = []
    
    graph_embeddings = torch.stack(graph_embeddings_list)
    boundary_embeddings = torch.stack(boundary_graph_embeddings_list)
    
    num_samples = min(len(graph_embeddings_list), len(boundary_graph_embeddings_list))
    shuffle_idx = torch.randperm(num_samples)
    
    for idx in range(num_samples):
        g1 = graph_embeddings[shuffle_idx[idx]]
        
        if len(boundary_graph_embeddings_list) > 1:
            g2 = boundary_embeddings[shuffle_idx[idx]]
        else:
            g2 = boundary_embeddings[0]
            
        dist = torch.norm(g1 - g2, p=2)
        new_batch = []
        
        for lmbd in np.linspace(0, 1.0, num=num_points):
            new_batch.append(g1 * lmbd + g2 * (1 - lmbd))
        new_batch = torch.stack(new_batch)
        
        y_new_batch = model_scoring_function(embeds=new_batch)['probs'].T
        thickness.append(dist.item() * len(np.where(gamma > y_new_batch[c1, :] - y_new_batch[c2, :])[0]) / num_points)
    
    return np.mean(thickness)


def calculate_boundary_complexity(final_embeds, D):
    """
    Takes a final embedding matrix for an adjacent class pair and calculates complexity.

    Args:
        final_embeds (torch.Tensor): The embedding matrix for an adjacent class pair. X_{c1 || c2|}
        D (int): Dimensionality factor for normalization.

    Returns:
        float: The calculated complexity value.
    """
    covariance_matrix = torch.cov(final_embeds)
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix).abs()
    normalized_eigenvalues = eigenvalues / eigenvalues.sum()
    entropy = -torch.sum(normalized_eigenvalues * torch.log(normalized_eigenvalues + 1e-8)).item()
    log_D = np.log(D)
    complexity = entropy / log_D
    return complexity
