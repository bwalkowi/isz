import random

import numpy as np
import pandas as pd
import networkx as nx
import ndlib
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics.SIRModel as sir
import ndlib.models.epidemics.SEISModel as seis
import ndlib.models.epidemics.SEIRModel as seir
import ndlib.models.epidemics.SWIRModel as swir
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers


PE = 0.1

PFS = 0.01
PFZ = 0.005


def plot_trend(model, iterations):
    trends = model.build_trends(iterations)
    viz = DiffusionTrend(model, trends)
    p = viz.plot()
    plt.show()


def sim_epidemics(graph, model, niterations, dynamic=True):
    nodes = set(graph.nodes())
    iterations = []
    state = {}

    for i in range(niterations):
        print('Iteration: {}'.format(i))

        it = model.iteration()
        iterations.append(it)
        state.update(it['status'])

        if dynamic:
            random_nodes = set(random.sample(nodes, int(len(nodes)*PFS)))
            random_nodes |= set(it['status'].keys())

            for u in random_nodes:
                neighbours = set(graph.adj[u])

                if state[u] == 0:

                    # try to find someone and make contact
                    [v] = random.sample(nodes - neighbours, 1)
                    if state[v] == 0 and PFS > random.uniform(0, 1):
                        graph.add_edge(u, v)
                    elif state[v] > 0 and PFZ > random.uniform(0, 1):
                        graph.add_edge(u, v)

                elif state[u] == 1:
                    # friends that suspect you are infected breaks ties with you
                    for v in neighbours:
                        if state[v] != 1 and PE > random.uniform(0, 1):
                            graph.remove_edge(u, v)

    plot_trend(model, iterations)


def sim_epidemics_with_animation(graph, model, nframes, dynamic=True):
    fig, ax = plt.subplots(figsize=(16,14))
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    pos = nx.spring_layout(graph)
    nodes = set(graph.nodes())
    iterations = []
    state = {}

    def update(nframe):
        ax.clear()

        it = model.iteration()
        iterations.append(it)
        state.update(it['status'])

        nx.draw_networkx_nodes(graph, pos, ax=ax, nodelist=list(state.keys()), node_size=30, node_color=list(state.values()), cmap=plt.cm.brg_r)
        nx.draw_networkx_edges(graph, pos, ax=ax)

        if dynamic:
            for u, u_state in state.items():# 
                if u_state == 0:
                    neighbours = set(graph.adj[u])
                    
                    # break ties with all your friends you suspect are infected
                    for v in neighbours:
                        if state[v] == 1 and PE > random.uniform(0, 1):
                            graph.remove_edge(u, v)

                    # try to find someone and make contact
                    [v] = random.sample(nodes - neighbours, 1)
                    if state[v] == 0 and PFS > random.uniform(0, 1):
                        graph.add_edge(u, v)
                    elif state[v] > 0 and PFZ > random.uniform(0, 1):
                        graph.add_edge(u, v)

    ani = FuncAnimation(fig, update, frames=nframes, interval=10, repeat=False)
    # ani.save(f'ani2.mp4', writer=writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800))
    plt.show()
    plot_trend(model, iterations)


def get_erdos_graph(g):
    return nx.erdos_renyi_graph(len(g), 0.15)


def get_watts_graph(g):
    return nx.watts_strogatz_graph(len(g), 3, 0.1)


def get_barabasi_graph(g):
    return nx.barabasi_albert_graph(len(g), 3)


def create_sir_model(graph, beta=0.01, gamma=0.01, infected=0.01):
    """    
    Susceptible     0
    Infected        1
    Removed         2

    beta - probability of transition from S to I
    gamma - probability of transition from I to R
    infected - part of population infected at the begining
    """
    model = sir.SIRModel(graph)

    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)
    cfg.add_model_parameter('gamma', gamma)
    cfg.add_model_parameter('percentage_infected', infected)
    model.set_initial_status(cfg)

    return model


def create_seis_model(graph, beta=0.01, gamma=0.005, alpha=0.05, infected=0.05):
    """
    Susceptible    0
    Infected 	   1
    Exposed 	   2

    beta - probability of transition from S to E
    gamma - probability of transition from E to I
    alpha - probability of transition from I to S
    infected - part of population infected at the begining
    """
    model = seis.SEISModel(graph)

    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)
    cfg.add_model_parameter('lambda', gamma)
    cfg.add_model_parameter('alpha', alpha)
    cfg.add_model_parameter("percentage_infected", infected)
    model.set_initial_status(cfg)

    return model


def create_seir_model(graph, beta=0.01, gamma=0.005, alpha=0.05, infected=0.05):
    """
    Susceptible    0
    Infected 	   1
    Exposed 	   2
    Removed 	   3

    beta - probability of transition from S to E
    gamma - probability of transition from E to I
    alpha - probability of transition from I to S
    infected - part of population infected at the begining
    """
    model = seir.SEIRModel(g)

    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', beta)
    cfg.add_model_parameter('gamma', gamma)
    cfg.add_model_parameter('alpha', alpha)
    cfg.add_model_parameter("percentage_infected", infected)
    model.set_initial_status(cfg)

    return model


def create_swir_model(graph, kappa=0.1, mu=0.5, nu=0.05, infected=0.05):
    """
    Susceptible     0
    Infected        1
    Weakened        2
    Removed         3

    kappa - probability of transition from S to I
    mu - probability of transition from S to W
    nu - probability of transition from W to I
    infected - part of population infected at the begining
    """
    model = swir.SWIRModel(graph)

    # Model Configuration
    cfg = mc.Configuration()
    cfg.add_model_parameter('kappa', kappa)
    cfg.add_model_parameter('mu', mu)
    cfg.add_model_parameter('nu', nu)
    cfg.add_model_parameter("percentage_infected", infected)
    model.set_initial_status(cfg)

    return model


graph = nx.read_gml("./bfmaier_anonymized_fb_network.gml")
# graph = nx.read_edgelist("./dictionary28.mtx")
model = create_seis_model(graph)

sim_epidemics_with_animation(graph, model, 400, dynamic=True)
# sim_epidemics(graph, model, 400, dynamic=False)
