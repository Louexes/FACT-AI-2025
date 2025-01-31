import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import networkx as nx
import copy
import secrets
import os
import pickle
import glob
import torch.nn.functional as F
import torch_geometric as pyg

from BoundaryTests import *


# New imports
import time
import seaborn as sns

# TODO: refactor
# from .datasets import *


class Trainer:
    def __init__(self,sampler,discriminator,criterion,scheduler,optimizer,dataset, seed, classes, budget_penalty=None):
        self.sampler = sampler
        self.discriminator = discriminator
        self.criterion = criterion
        self.budget_penalty = budget_penalty
        self.scheduler = scheduler
        self.optimizer = optimizer if isinstance(optimizer, list) else [optimizer]
        self.dataset = dataset
        self.iteration = 0
        self.seed = seed #added
        self.classes = classes #added

    def init(self):
        self.sampler.init()
        self.iteration = 0

    def train(self, iterations,
              show_progress=True,
              target_probs: dict[int, tuple[float, float]] = None,
              target_size=None,
              w_budget_init=1,
              w_budget_inc=1.05,
              w_budget_dec=0.99,
              k_samples=32,
              log_file=None): #added:
        # self.bkup_state = copy.deepcopy(self.sampler.state_dict())
        # self.bkup_criterion = copy.deepcopy(self.criterion)
        # self.bkup_iteration = self.iteration

        start_time = time.time()

        self.discriminator.eval()
        self.sampler.train()
        budget_penalty_weight = w_budget_init
        for _ in (bar := tqdm(
            range(int(iterations)),
            initial=self.iteration,
            total=self.iteration+iterations,
            disable=not show_progress
        )):
            for opt in self.optimizer:
                opt.zero_grad()
            cont_data = self.sampler(k=k_samples, mode='continuous')
            disc_data = self.sampler(k=1, mode='discrete', expected=True)
            # TODO: potential bug
            cont_out = self.discriminator(cont_data, edge_weight=cont_data.edge_weight)
            disc_out = self.discriminator(disc_data, edge_weight=disc_data.edge_weight)
            if target_probs and all([
                min_p <= disc_out["probs"][0, classes].item() <= max_p
                for classes, (min_p, max_p) in target_probs.items()
            ]):
                if target_size and self.sampler.expected_m <= target_size:
                    pred_probs = disc_out["probs"].mean(axis=0).tolist()
                    print("Stopped:", pred_probs)
                    break
                budget_penalty_weight *= w_budget_inc
            else:
                budget_penalty_weight = max(w_budget_init, budget_penalty_weight * w_budget_dec)

            loss = self.criterion(cont_out | self.sampler.to_dict())
            if self.budget_penalty:
                loss += self.budget_penalty(self.sampler.theta) * budget_penalty_weight
            loss.backward()  # Back-propagate gradients

            for opt in self.optimizer:
                opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # logging
            size = self.sampler.expected_m
            scores = disc_out["logits"].mean(axis=0).tolist()
            score_dict = {v: scores[k] for k, v in self.dataset.GRAPH_CLS.items()}
            penalty_weight = {'bpw': budget_penalty_weight} if self.budget_penalty else {}
            bar.set_postfix({'size': size} | penalty_weight | score_dict)
            # print(f"{iteration=}, loss={loss.item():.2f}, {size=}, scores={score_dict}")
            self.iteration += 1
        else:
            return False, self.iteration
        return True, self.iteration

    @torch.no_grad()
    def predict(self, G): 
        batch = pyg.data.Batch.from_data_list([self.dataset.convert(G, generate_label=True)])
        return self.discriminator(batch)

    @torch.no_grad()
    def quantatitive(self, gnninterpreter_class, seed, pair, log_file_analysis, sample_fn=None):

        print("Running new func")
        sample_fn = lambda: self.evaluate(bernoulli=True)

        boundary_graph_embeddings = []
        probs = []
        final_embeddings = torch.empty((0, self.predict(sample_fn()[0])['embeds_last'].shape[1]))

        print("Sampling 500 Boundary Graphs")
        for i in range(500):
            probs.append(self.predict(sample_fn()[0])["probs"][0].numpy().astype(float))
            boundary_graph_embeddings.append(self.predict(sample_fn()[0])['embeds'])
            final_embed = self.predict(sample_fn()[0])['embeds_last']
            final_embeddings = torch.vstack([final_embeddings, final_embed])
            
        
        boundary_margin1 = calculate_boundary_margin(boundary_graph_embeddings, gnninterpreter_class[pair[0]]) #Both inputs are lists where each entry in list is embedding of a graph (i.e a tensor)
        boundary_thickness1 = boundary_thickness(gnninterpreter_class[pair[0]], boundary_graph_embeddings, self.discriminator, pair[0], pair[1]) #Both inputs are lists where each entry in list is embedding of a graph (i.e a tensor)

        #we compute margin and thickness twice, with the other class from gnninterpreter to reproduce the matrices in Figure 3 of the original paper.
        boundary_margin2 = calculate_boundary_margin(boundary_graph_embeddings, gnninterpreter_class[pair[1]]) #Both inputs are lists where each entry in list is embedding of a graph (i.e a tensor)
        boundary_thickness2 = boundary_thickness(gnninterpreter_class[pair[1]], boundary_graph_embeddings, self.discriminator, pair[0], pair[1]) #Both inputs are lists where each entry in list is embedding of a graph (i.e a tensor)

        boundary_complexity = calculate_boundary_complexity(final_embeddings, D = 6)

        print(f"{boundary_margin1=}, {boundary_margin2=}, {boundary_thickness1=}, {boundary_thickness2=}, {boundary_complexity=}") 

        log_file_analysis.write(f"{seed}\t{pair}\t{np.mean(probs, axis=0)}\t{np.std(probs, axis=0)}\t{boundary_margin1}\t{boundary_margin2}\t{boundary_thickness1}\t{boundary_thickness2}\t{boundary_complexity}\n")

        return float(boundary_margin1), float(boundary_margin2), float(boundary_thickness1), float(boundary_thickness2)
    

    @torch.no_grad() #NOTE THIS WAS ADDED FOR GNNINTERPRETER. We want to sample 500 graphs for each class for boundary analysis.
    def quantatitiveInterpreter(self, sample_size=500, sample_fn=None):
        sample_fn = lambda: self.evaluate(bernoulli=True)
        embeds = []
        probs = []
        for i in range(500):
            graph = sample_fn()[0]
            pred = self.predict(graph)
            embeds.append(pred["embeds"])
            probs.append(pred["probs"][0].numpy().astype(float))
        mean_probs = np.mean(probs, axis=0)
        std_probs = np.std(probs, axis=0)
        return embeds, mean_probs, std_probs
    
    @torch.no_grad()
    def quantatitive_baseline(self, **kwargs):
        return self.quantatitive(sample_fn=lambda: nx.gnp_random_graph(n=self.sampler.n, p=1/self.sampler.n),
                                 **kwargs)

    # TODO: do not rely on dataset for drawing
    @torch.no_grad()
    def evaluate(self, *args, show=False, connected=False, **kwargs):
        self.sampler.eval()
        G = self.sampler.sample(*args, **kwargs)
        probs = None

        if G is None or G.number_of_nodes() == 0:
            return (G, probs)
        
        if connected:
            G = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda g: g.number_of_nodes())[-1]
            G = sorted([G.subgraph(c) for c in nx.connected_components(G)], key=lambda g: g.number_of_nodes())[-1]

        if show:
            probs = self.show(G)
            plt.show()
        else:
            probs = self.predict(G)["probs"].mean(dim=0).tolist()

        return (G, probs)

    def show(self, G, ax=None):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        pred = self.predict(G)
        print(f"{pred=}")
        logits = pred["logits"].mean(dim=0).tolist()
        probs = pred["probs"].mean(dim=0).tolist()
        print(f"{n=} {m=}")
        print(f"{logits=}")
        print(f"{probs=}")
        self.dataset.draw(G, ax=ax)

        return probs

    def save_graph(self, G, cls_idx, root="results"):
        if isinstance(cls_idx, tuple):
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx[0]]}-{self.dataset.GRAPH_CLS[cls_idx[1]]}"
        else:
            path = f"{root}/{self.dataset.name}/{self.dataset.GRAPH_CLS[cls_idx]}"
        name = secrets.token_hex(4).upper() # TODO: use hash of the graph to avoid duplicate
        os.makedirs(path, exist_ok=True)
        pickle.dump(G, open(f"{path}/{name}.pkl", "wb"))
        self.show(G)
        plt.savefig(f"{path}/{name}.png", bbox_inches="tight")
        plt.show()

    def load_graph(self, id, root="results"):
        path = f"{root}/{self.dataset.name}/*"
        G = pickle.load(open(glob.glob(f"{path}/{id}.pkl")[0], "rb"))
        self.show(G)
        return G

    def save_sampler(self, cls_idx, root="sampler_ckpts"):
        if isinstance(cls_idx, int):
            path = f"{root}/{self.dataset.name}/{cls_idx}"
        else:
            path = f"{root}/{self.dataset.name}/{'-'.join(map(str, cls_idx))}"
        name = secrets.token_hex(4).upper() # TODO: use hash of the graph to avoid duplicate
        os.makedirs(path, exist_ok=True)
        self.sampler.save(f"{path}/{name}.pt")

    def batch_generate(self, cls_idx, total, epochs, show_progress=True):
        pbar = tqdm(total=total)
        count = 0
        while count < total:
            self.init()
            if self.train(epochs, show_progress=show_progress):
                self.save_sampler(cls_idx)
                count += 1
                pbar.update(1)

    def get_training_success_rate(self, total, epochs, show_progress=False):
        iters = []
        for _ in (bar := trange(total)):
            self.init()
            if self.train(epochs, show_progress=show_progress)[0]:
                iters.append(self.iteration)
            bar.set_postfix({'count': len(iters)})
        return iters
    