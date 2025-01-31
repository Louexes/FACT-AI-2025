import networkx as nx
import pandas as pd
import torch_geometric as pyg

from gnn_xai_common.datasets import BaseGraphDataset
from gnn_xai_common.datasets.utils import default_ax, unpack_G


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
        0 : "No-Enzyme",
        1: "Enzyme"}

    def __init__(self, *,
                 name='PROTEINS',
                 url='https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip',
                 **kwargs):
        self.url = url
        super().__init__(name=name, **kwargs)

    @property
    def raw_file_names(self):
        return ["PROTEINS_A.txt",
                "PROTEINS_graph_indicator.txt",
                "PROTEINS_graph_labels.txt",
                "PROTEINS_node_attributes.txt",
                "PROTEINS_node_labels.txt"]

    def download(self):
        pass

    def generate(self):
        edges = pd.read_csv(self.raw_paths[0], header=None).to_numpy(dtype=int) - 1
        graph_idx = pd.read_csv(self.raw_paths[1], header=None)[0].to_numpy(dtype=int) - 1
        graph_labels = pd.read_csv(self.raw_paths[2], header=None)[0].to_numpy(dtype=int) - 1
        # edge_labels = pd.read_csv(self.raw_paths[3], header=None)[0].to_numpy(dtype=int)
        node_labels = pd.read_csv(self.raw_paths[4], header=None)[0].to_numpy(dtype=int) - 1
        super_G = nx.Graph(edges.tolist(), label=graph_labels)
        nx.set_node_attributes(super_G, dict(enumerate(node_labels)), name='label')
        nx.set_node_attributes(super_G, dict(enumerate(graph_idx)), name='graph')
        # nx.set_edge_attributes(super_G, dict(zip(zip(*edges.T), edge_labels)), name='label')
        return unpack_G(super_G)

    # TODO: use EDGE_WIDTH
    @default_ax
    def draw(self, G, pos=None, label=False, ax=None):
        pos = pos or nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               ax=ax,
                               nodelist=G.nodes,
                               node_size=500,
                               node_color=[
                                   self.NODE_COLOR[G.nodes[v]['label']]
                                   for v in G.nodes
                               ])
        if label:
            nx.draw_networkx_labels(G, pos,
                                    ax=ax,
                                    labels={
                                        v: self.NODE_CLS[G.nodes[v]['label']]
                                        for v in G.nodes
                                    })
        nx.draw_networkx_edges(G.subgraph(G.nodes), pos, ax=ax, width=6)

    def process(self):
        super().process()

