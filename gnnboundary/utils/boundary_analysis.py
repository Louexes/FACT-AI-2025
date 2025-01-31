from collections import namedtuple
import random

import numpy as np

import torch 

import torch_geometric as pyg


def boundary_analysis(model, dataset_1, dataset_2, key='embeds_last', k=100, n=100):
    BoundaryAnalysisResult = namedtuple("BoundaryAnalysisResult",[
        "adj_ratio",
        "interp_matrix",
        "bound_results"
    ])
    is_connected = []
    interp_matrix = []
    bound_results = []

    embeds_1 = custom_transform(model, dataset_1, key=key)
    embeds_2 = custom_transform(model, dataset_2, key=key)

    #embeds_1 = dataset_1.model_transform(model, key=key)
    #embeds_2 = dataset_2.model_transform(model, key=key)
    
    for _ in range(k):
        sample1 = random.choice(embeds_1)
        sample2 = random.choice(embeds_2)
        interp = []
        min_diff, bound_result = 1, None
        for i in range(1, n):
            result = model(**{key: (sample2 - sample1) * i / n + sample1})
            diff = result['probs'].sort(descending=True)[0][:2].diff().abs().item()
            if diff < min_diff:
                min_diff = diff
                bound_result = result
            interp.append(result["logits"].argmax().item())
        interp_matrix.append(interp)
        bound_results.append(bound_result)
        is_connected.append(np.unique(interp).shape[0] <= 2)
    return BoundaryAnalysisResult(
        adj_ratio=np.mean(is_connected),
        interp_matrix=interp_matrix,
        bound_results=bound_results
    )


def custom_transform(model, dataset, key = 'embeds_last'):
    """
    Custom logic to replace model_transform to get embeddings.

    This function processes a dataset of graphs through a given model and 
    extracts embeddings specified by the key. The embeddings from each graph 
    are concatenated into a single tensor.

    Args:
        model (torch.nn.Module): The model used to process the graphs.
        dataset (iterable): A collection of graph data structures to be processed.
        key (str, optional): The key to extract specific embeddings from the model's output. 
                            Defaults to 'embeds_last'.

    Returns:
        torch.Tensor: A tensor containing the concatenated embeddings from all graphs in the dataset.
    """
    embeds = []
    model.eval()
    for graph in dataset:
        embeds.append(model(graph)[key])
    return torch.cat(embeds, dim=0)


def custom_transform2(model, dataset, batch_size = 32, key = 'embeds_last'):
    """
    Custom logic to replace model_transform to get embeddings.

    This function processes a dataset of graphs through a given model and 
    extracts embeddings specified by the key. The embeddings from each graph 
    are concatenated into a single tensor.

    Args:
        model (torch.nn.Module): The model used to process the graphs.
        dataset (iterable): A collection of graph data structures to be processed.
        key (str, optional): The key to extract specific embeddings from the model's output. 
                            Defaults to 'embeds_last'.

    Returns:
        torch.Tensor: A tensor containing the concatenated embeddings from all graphs in the dataset.
    """
    embeds = []
    model.eval()
    for batch in pyg.data.DataLoader(dataset, batch_size=batch_size, shuffle=False):
        embeds.append(model(batch)[key])
    return torch.cat(embeds, dim=0)
