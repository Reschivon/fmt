import math
from typing import List

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from pqdict import pqdict


class Node:
    def __init__(self, x, v=None, d=None):
        self.x = x  # position
        self.v = v  # direction vector
        self.d = d  # frontier radius
        self.cost = float('inf')  # cost f(x)


class FMTWosPlanner():
    def __init__(
        self,
        map_design: np.ndarray,
        n_samples: int = 1000,
        r_n: float = 20.0,
        path_resolution: float = 0.1,
        rr: float = 1.0,
        max_search_iter: int = 10000,
        seed: int = 1,
    ):
        """
        Fast Marching Tree Path Planner with Frontiers

        Args:
            map_design (np.ndarray): Obstacle map described by a binary image. 1: free nodes; 0: obstacle nodes
            n_samples (int, optional): Number of nodes to sample. Defaults to 1000.
            r_n (float, optional): Range to find neighbor nodes. Defaults to 20.0.
            path_resolution (float, optional): Resolution of paths to check collisions. Defaults to 0.1.
            rr (float, optional): Distance threshold to check collisions. Defaults to 1.0.
            max_search_iter (int, optional): Number of maximum iterations. Defaults to 10000.
            seed (int, optional): Random seed. Defaults to 0.
        """

        # hyperparameters
        self.map_size = map_design.shape
        self.path_resolution = path_resolution
        self.rr = rr
        self.n_samples = n_samples
        self.r_n = r_n
        self.max_search_iter = max_search_iter
        self.prng = np.random.RandomState(seed)  # initialize PRNG

        # construct obstacle tree
        obstacles = np.argwhere(map_design == 0)
        self.obstacles_tree = cKDTree(obstacles)

        # initialize graph
        self.graph = nx.Graph()
        self.node_list: List[Node] = list()
        i = 0
        while len(self.node_list) < self.n_samples:
            node = self.prng.uniform(0, self.map_size)
            if self.check_collision(node, None):
                new_node = Node(x=node)
                self.node_list.append(new_node)
                self.graph.add_node(i)
                i += 1

    def plan(self,
             start: np.ndarray,
             goal: np.ndarray,
             heuristic_weight: int = 0.0) -> dict:
        """
        Run path planning

        Args:
            start (np.ndarray): Start location
            goal (np.ndarray): Goal location
            heuristic_weight (int, optional): Weight for Euclidean heuristics. Defaults to 0.0.

        Returns:
            dict: Containing path, number of steps required, and goal flag
        """
        start = np.asarray(start)
        goal = np.asarray(goal)
        assert self.check_collision(start, None)
        assert self.check_collision(goal, None)

        # Remove existing edges
        self.graph.remove_edges_from(list(self.graph.edges))

        # Create Node instances for start and goal
        start_node = Node(x=start)
        goal_node = Node(x=goal)
        start_node.cost = 0  # set cost of start node to 0
        self.node_list.append(start_node)
        start_id = len(self.node_list) - 1
        self.graph.add_node(start_id)
        self.node_list.append(goal_node)
        goal_id = len(self.node_list) - 1
        self.graph.add_node(goal_id)

        # Build KDTree of node positions
        node_positions = np.array([node.x for node in self.node_list])
        node_tree = cKDTree(node_positions)

        # Initialize unvisited set and priority queue (heap)
        V_unvisited = set(range(len(self.node_list)))
        V_unvisited.remove(start_id)

        H = pqdict({start_id: self.node_list[start_id].cost})

        # Initialize goal_flag
        goal_flag = 0

        # Start expansion loop
        n_steps = 0
        while H:
            # Pop x_min from H
            x_min_id = H.pop()
            x_min = self.node_list[x_min_id]

            # If x_min is goal node, break
            if x_min_id == goal_id:
                print("Reached goal")
                goal_flag = 1
                break

            # Compute c_min = x_min.x - x_min.d * x_min.v
            if x_min.d is not None and x_min.v is not None:
                c_min = x_min.x - x_min.d * x_min.v
            else:
                # For the start node, set c_min = x_min.x
                c_min = x_min.x

            # Find neighbors within radius r_n of x_min.x
            N_x_min = node_tree.query_ball_point(x_min.x, self.r_n)

            # For each x_near in N_x_min
            for x_near_id in N_x_min:
                if x_near_id == x_min_id:
                    continue
                
                if x_near_id not in V_unvisited:
                    continue
                
                x_near = self.node_list[x_near_id]
                # Compute distance from c_min to x_near.x
                d_cmin_xnear = np.linalg.norm(c_min - x_near.x)
                # Check if x_near.x is beyond x_min.d from c_min
                d_min = x_min.d if x_min.d is not None else 0
                if d_cmin_xnear <= d_min:
                    continue  # Skip if within frontier
                # Check collision between c_min and x_near.x
                if self.check_collision(c_min, x_near.x):
                    # Collision-free
                    # Compute new cost
                    cost_cmin_xmin = d_min  # Since cost(c_min, x_min.x) = x_min.d
                    cost_cmin_xnear = d_cmin_xnear
                    new_cost = x_min.cost - cost_cmin_xmin + cost_cmin_xnear
                    x_near.cost = new_cost
                    
                    # Update x_near.v and x_near.d
                    vec = x_near.x - c_min
                    x_near.v = vec / np.linalg.norm(vec)
        
                    x_near.d = d_cmin_xnear
                    # Add x_near to H
                    H.additem(x_near_id, x_near.cost)
                    # Remove x_near_id from V_unvisited
                    V_unvisited.remove(x_near_id)
                    # Add edge from x_min to x_near
                    self.graph.add_edge(x_min_id, x_near_id)
                else:
                    # Collision between c_min and x_near.x
                    # Check collision between x_min.x and x_near.x
                    if not self.check_collision(x_min.x, x_near.x):
                        # Both paths are blocked
                        continue
                    else:
                        # Path from x_min.x to x_near.x is collision-free
                        # Compute cost
                        cost_xmin_xnear = np.linalg.norm(x_min.x - x_near.x)
                        new_cost = x_min.cost + cost_xmin_xnear
                        x_near.cost = new_cost
                        
                        # Update x_near.v and x_near.d
                        vec = x_near.x - x_min.x
                        x_near.v = vec / np.linalg.norm(vec)
                        
                        x_near.d = cost_xmin_xnear
                        # Add x_near to H
                        H.additem(x_near_id, x_near.cost)
                        # Remove x_near_id from V_unvisited
                        V_unvisited.remove(x_near_id)
                        # Add edge from x_min_id to x_near_id
                        self.graph.add_edge(x_min_id, x_near_id)

            n_steps += 1
            if n_steps >= self.max_search_iter:
                print("Reached maximum number of iterations")
                break

        if goal_flag:
            # Extract path
            path_indices = nx.shortest_path(self.graph, start_id, goal_id)
            path = np.array([self.node_list[idx].x for idx in path_indices])
        else:
            print("Search failed")
            path = None

        return {
            "path": path,
            "n_steps": n_steps,
            "goal_flag": goal_flag,
        }

    def check_collision(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """
        Check collision

        Args:
            src (np.ndarray): Source node
            dst (np.ndarray): Destination node

        Returns:
            bool: True if no collisions were found and False otherwise
        """
        pr = self.path_resolution
        if (dst is None) or np.all(src == dst):
            return self.obstacles_tree.query(src)[0] > self.rr

        dx, dy = dst[0] - src[0], dst[1] - src[1]
        yaw = math.atan2(dy, dx)
        d = math.hypot(dx, dy)
        steps = np.arange(0, d, pr).reshape(-1, 1)
        pts = src + steps * np.array([math.cos(yaw), math.sin(yaw)])
        pts = np.vstack((pts, dst))
        return bool(self.obstacles_tree.query(pts)[0].min() > self.rr)
