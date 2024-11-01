import fmt
import utils
import fmt_wos

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from fmt_wos import FMTWosPlanner

import signal
# ctrl c
signal.signal(signal.SIGINT, signal.SIG_DFL)

def visualize_result2(map_design: np.ndarray, planner: FMTWosPlanner,
                     path_info: dict) -> None:
    plt.figure()
    plt.imshow(map_design, cmap="gray")
    # print("node_list", [node.x for node in planner.node_list])
    nx.draw(planner.graph, [ (node.x[1], node.x[0]) for node in planner.node_list],
            node_size=1,
            alpha=.5)
    path = path_info["path"]
    plt.plot(path[:, 1], path[:, 0], 'r-', lw=2)
    plt.gcf().set_size_inches(15, 12)
    plt.show()
    
map_design = utils.load_map_design("map.png", size=[400, 200])

# planner = fmt.FMTPlanner(map_design, n_samples=1000, r_n=20, path_resolution=0.1, rr=1.0, max_search_iter=10000)
planner = fmt_wos.FMTWosPlanner(map_design, n_samples=1000, r_n=20, path_resolution=0.1, rr=1.0, max_search_iter=10000)

path_info = planner.plan([180, 20], [20, 380])
visualize_result2(map_design, planner, path_info)

# using Euclidean heuristics
path_info = planner.plan([180, 20], [20, 380], heuristic_weight=1.0)
visualize_result2(map_design, planner, path_info)
