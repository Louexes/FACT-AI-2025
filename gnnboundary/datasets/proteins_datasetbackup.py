import networkx as nx
import pandas as pd
import torch_geometric as pyg

from datasets import load_dataset
from torch_geometric.data import Data
from gnn_xai_common.datasets import BaseGraphDataset

import torch

import random


class PROTEINSDataset(BaseGraphDataset):
    NODE_CLS = {
        0: "0", 
        1: "1"
        }
    NODE_COLOR = {
        0: "red", 
        1: "blue"
        }
    GRAPH_CLS = {
        0: "Not Protein",
        1: "Protein"}

    def __init__(self, name="PROTEINS", url=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.url = url

    def generate(self):
        try:
            dataset_hf = load_dataset("graphs-datasets/PROTEINS")
            data_list = []
            for graph in dataset_hf["train"]:
    
                data_list.append(
                    Data(
                        x=torch.tensor(graph["node_feat"], dtype=torch.float),
                        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                        y=torch.tensor([graph["y"]], dtype=torch.long),
                    )
                )
            return data_list
        except Exception as e:
            raise RuntimeError(f"Failed to load PROTEINS dataset: {str(e)}")
        

    def convert(self, G, generate_label=False):
        if isinstance(G, list):
            return pyg.data.Batch.from_data_list([self.convert(g) for g in G])
        if isinstance(G, nx.Graph):
            G = nx.convert_node_labels_to_integers(G)
            node_labels = [G.nodes[i]['label']
                        if 'label' in G.nodes[i] or not generate_label
                        else random.choice(list(self.NODE_CLS))
                        for i in G.nodes]
            if G.number_of_edges() > 0:
                if hasattr(self, "EDGE_CLS"):
                    edge_labels = [G.edges[e]['label']
                                if 'label' in G.edges[e] or not generate_label
                                else random.choice(list(self.EDGE_CLS))
                                for e in G.edges]
                    edge_index, edge_attr = pyg.utils.to_undirected(
                        torch.tensor(list(G.edges)).T,
                        torch.eye(len(self.EDGE_CLS))[edge_labels].float(),
                    )
                else:
                    edge_index, edge_attr = pyg.utils.to_undirected(
                        torch.tensor(list(G.edges)).T,
                    ), None
            else:
                if hasattr(self, "EDGE_CLS"):
                    edge_index, edge_attr = torch.empty(2, 0).long(), torch.empty(0, len(self.EDGE_CLS))
                else:
                    edge_index, edge_attr = torch.empty(2, 0).long(), None
            return pyg.data.Data(
                G=G,
                x=torch.eye(len(self.NODE_CLS))[node_labels].float(),
                y=torch.tensor(G.graph['label'] if "label" in G.graph else -1).long(),
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
        elif isinstance(G, pyg.data.Data):
            return G
        else:
            raise TypeError("Input must be a NetworkX graph or a PyTorch Geometric Data object")

