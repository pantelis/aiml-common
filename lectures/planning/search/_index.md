---
title: Planning with Search 
weight: 95
draft: false
---

# Planning with Search 

In [classical planning]({{<ref "../classical-planning">}}) we have seen that symbolic representations enriched with propositional logic in the Knowledge Base allow us to reason and come up with a sequence of actions that the agent needs to execute to reach the goal. This section provides the background that _connects_ the domain-independent planner that solves the planning problem to _search_. To do so, it is instructive to look initially to problems that are simpler in terms of state and action representation. We effectively zoom out from the  _factored_ representation of the environment's state and treat it as _atomic_ i.e. not broken down into its individual variables. 

Atomic state representations of an environment are adequate for a a variety of global planning tasks:  one striking use case is path planning There, the scene or environment takes the form of a global map and the goal is to move the embodied agent from a starting state to a goal state. If we assume that the global map takes the form of a grid with a suitable resolution, each grid tile (or cell) represents a different atomic state than any other cell. Similar considerations can be made for other forms of the map e.g. a graph form. 

Given such state representation, we will use _search_ to find the action sequence that the agent must produce to reach a goal state. Note that in most cases, we are dealing with _informed_ rather than _blind_ search, where we are also given task-specific knowledge (that we use to develop suitable heuristics) to find the solution as we will see shortly. 

## Forward-Search

We will develop the algorithm for the task at hand which is to find the path between a starting state and the goal state in a map. Not just any path but the _minimum cost_ path when the state transition graph is revealed incrementally through a series of actions and associated individual costs (cost function). The task is depicted below. 

![path-finding](images/parking-lot.png#center)
*A map of a parking lot as determined via postprocessing LIDAR returns. Obstacles colored in yellow are tall obstacles, brown obstacles are curbs, and green obstacles are tree branches that are of no relevance to ground navigation.*

In practice, maps likes the one above are local both in terms of space (each snapshot is relative to the location of the agent) as well as in terms of time (at the time the agent took these measurements). We can take any map like the one above and form its discrete equivalent such as shown below. We usually call this type _metric map_ and for the purposes of this chapter this is our search area and in it lie all possible feasible _solutions_, each is a _sequence of actions_ distinguished by _path cost_. In this example, the least cost path is actually the geographically longer path - the line thickness for each of the two possible solutions in this figure is proportional to its cost. 

![cost-definition](images/cost-definition.png#center)
*An example search area with action-cost function where left turns are penalized compared to right.This is common penalization in path planning for delivery vehicles.*

The alternative map representation is _topological_ where the environment is represented as a graph where nodes indicated significant grounded features and edges denote topological relationships (position, orientation, proximity etc.). The sometimes confusing part is that irrespectively of metric or topological representations, the _forward search_ methods we look in this chapter all function on _graphs_ given an initial state $s_I$ and the goal state $s_G$ that is reached after potentially a finite number of actions (if a solution exist). The following pseudo-code is from [Steven LaValle's book - Chapter 2](http://planning.cs.uiuc.edu/)

<pre id="forward-search" style="display:hidden;">
    \begin{algorithm}
    \caption{Forward Search}
    \begin{algorithmic}
    \INPUT $s_I$, $s_G$
    \STATE Q.Insert($s_I$), mark $s_I$ as explored.
    \WHILE{Q not empty} 
        \STATE $s \leftarrow Q.GetFirst()$
        \IF{$s \in S_G$}
            \RETURN SUCCESS
        \ENDIF
        \FOR{$a \in A(s)$}
            \STATE $s' \leftarrow f(s,a)$
            \IF{$s'$ not visited}
               \STATE mark $s'$ as explored 
               \STATE Q.Insert($s'$)
            \ELSE 
                \STATE Resolve duplicate $s'$
            \ENDIF
        \ENDFOR
    \ENDWHILE
    \end{algorithmic}
    \end{algorithm}
</pre>

The forward search uses two data structures, a priority queue (Q) and a list and proceeds as follows:

1. Provided that the starting state is not the goal state, we add it to a priority queue called the  _frontier_ (also known as _open list_ in the literature but we avoid using this term as its implemented is a queue). The name frontier is synonymous to _unexplored_. 
2. We _expand_ each state in our frontier, by applying the finite set of actions, generating a new list of states. We use a list that we call the _explored set_ or _closed list_ to remember each node (state) that we expanded.  This is the list data structured we mentioned. 
3. We then go over each newly generated state and before adding it to the frontier we check whether it has been expanded before and in that case we discard it. 

## Forward-search approaches

The only significant difference between various search algorithms is the specific priority function that implements line 3: $s \leftarrow Q.GetFirst()$ in other words retrieves a state held in the priority queue for expansion. 

| Search     | Queue Policy    | Details    |
| --- | --- | --- |
| **Depth-first search (DFS)**   |  LIFO  | Search frontier is driven by aggressive exploration of the transition model. The algorithm makes deep incursions into the graph and retreats only when it run out of nodes to visit. It does not result in finding the shortest path to the goal. | 
|  **Breath-first search**  |   FIFO  | Search frontier is expanding uniformly like the propagation of waves when you drop a stone in water. It therefore visit vertices in increasing order of their distance from the starting point. |
|   **Dijkstra**  |  Cost-to-Come or Past-Cost  |     |
|   **A-star**   |  Cost-to-Go or Future-Cost |     |


### Depth-first search (DFS)

In undirected graphs, depth-first search answers the question: What parts of the graph are reachable from a given vertex. 
It also finds explicit paths to these vertices, summarized in its search tree as shown below.

![depth-first](images/depth-first.png#center)
*Depth-first can't find optimal (shortest) paths. Vertex C is reachable from S by traversing just one edge, while the DFS tree
shows a path of length 3. On the right the DFS search tree is shown assuming alphabetical order in breaking up ties. Can you explain the DFS search tree?*

DFS can be run verbatim on directed graphs, taking care to traverse edges only in their prescribed directions.

### Breadth-first search (BFS)

In BFS the lifting of the starting state $s$, partitions the graph into layers: $s$ itself (vertex at distance 0), the vertices at distance 1 from it, the vertices at distance 2 from it etc. 

![breadth-first](images/breadth-first.png#center)
*BFS expansion in terms of layers of vertices - each layer at increasing distance from the layer that follows*. 

![breadth-first-2](images/breadth-first-2.png#center)
*Queue contents during BFS and the BFS search tree assuming alphabetical order. Can you explain the BFS search tree? Is the BFS search tree a shortest-path tree?* 
 
#### Dijkstra's Algorithm

Breadth-first search finds shortest paths in any graph whose edges have unit length. Can we adapt it to a more general graph G = (V, E) whose edge lengths $l(e)$ are positive integers? These lengths effectively represent the cost of traversing the edge. fHere is a simple trick for converting G into something BFS can handle: break G’s long edges into unit-length pieces, by introducing “dummy” nodes as shown next.

![dijkstras-graph](images/dijkstras-graph.png#center)
*To construct the new graph $G'$ for any edge $e = (s, s^\prime)$ of $E$, replace it by $l(e)$ edges of length 1, by adding $l(e) − 1$
dummy nodes between nodes $s$ and $s^\prime$*. 

With the shown transformation, we can now run BFS on $G'$ and the search tree will reveal the shortest path of each goal node from the starting point. 

The transformation allows us to solve the problem but it did result in an inefficient search where most of the nodes involved are searched but definitely will never be goal nodes. To look for more efficient ways to absorb the edge length $l(e)$ we use the following of cost.  

![search-cost-definitions](images/search-cost-defintions.png#center)
*Cost-to-come($s$) or PastCost($s$) vs. Cost-to-Go($s$) or FutureCost($s$). PastCost($s$) is the minimum cost from the start state $s_I$ to state $s$. FutureCost($s$) is the minimum cost from the state $s$ to the goal state $s_G$. The PastCost is used as the prioritization metric of the queue in Dijkstra's algorithm. The addition of the PastCost with an estimate of the FutureCost, the heuristic $h(s)$, i.e. $G(s) =$ PastCost($s$)+$h(s)$, is used as the corresponding metric in the A\* algorithm. What would be the ideal metric?*

The following example is instructive of the execution steps of the algorithm. 

![dijkstras-example](images/dijkstras-example.png#center)
*Example of Dijkstra's algorithm execution steps and with $s_I=A$*

The exact same pseudo-code is executed as before but the priority metric $C(s^\prime)=C^*(s) + l(e)$ now accounts for costs as they are calculated causing the queue to be reordered accordingly.  Here, $C(s^\prime)$ represents the best cost-to-come that is known so far, but we do not write $ C^*$ because it is not yet known whether $ s^\prime$ was reached optimally. Due to this, some work is required in line 12: $\texttt{Resolve duplicate}$ $s^\prime$. If $s^\prime$ already exists in $ {Q}$, then it is possible that the newly discovered path to $s^\prime$ is more efficient. If so, then the cost-to-come value $ {C}(s^\prime)$ must be lowered for $s^\prime$, and $ {Q}$ must be reordered accordingly. When does $ {C}(s)$ finally become $ C^*(x)$ for some state $s$? Once $s$ is removed from $ {Q}$ using $ {Q}.GetFirst()$, the state becomes dead, and it is known that $ x$ cannot be reached with a lower cost. 

Using the [demo](https://qiao.github.io/PathFinding.js/visual/) link below, we can construct a wall-world where we have almost enclosed the starting state. We will comment on this result after treating the A* algorithm.

![dijkstras-demo](images/dijkstras-demo.png#center)
*Dijkstra's algorithm demo*

#### A* Algorithm

Dijkstra's algorithm is very much related to the _Uniform Cost Search_ algorithm and in fact logically they are equivalent as the algorithm explores uniformly all nodes that have the same PastCost. In the Astar algorithm, we start using the fact that we _know_ the end state and therefore attempt to find methods that bias the exploration towards it. 

As mentioned in the cost definition figure above, A* uses both $C^*(s)$  and an estimate of the optimal Cost-to-go or FutureCost $G^*(s)$ because obviously to know exactly $G^*(s)$ is equivalent to solving the original search problem.  Therefore the metric for prioritizing the Q queue is: 

$$C^*(s) + h(s)$$

If $h(s)$ is an underestimate of the $G^*(s)$ the Astar algorithm is guaranteed to fine optimal plans. 

For an example of a heuristic consider this problem:

![astar-example](images/astar-simple-example.png#center)
*A simple example showcasing a modified to what is described above priority metric. What we use is a modification of the edge cost $l'(e)=l(e) + [h(s^\prime)-h(s)]$. Is there a difference?*

In this example all $l(e)=1$ and the heuristic is a penalty from how much a transition to another node (an action) takes us away from the end state (adopted from [CS221](https://stanford-cs221.github.io/)). 

Using the interactive demo page below, repeating the same example wall-world, we can clearly see the substantial difference in search speed and the beamforming effect as soon as the wave (frontier) evaluates nodes where the heuristic (the Manhattan distance from the goal node) becomes dominant. Notice the difference with the UCS / Dijkstra's algorithm in the number of nodes that needed to be evaluated. 

![astar-demo](images/astar-demo.png#center)
*$A^\*$ algorithm demo

## Forward Search Algorithm Implementation

### Interactive Demo 

This [demo](https://qiao.github.io/PathFinding.js/visual/) is instructive of the various search algorithms we covered here. You can introduce using your mouse obstacles in the canvas and see how the various search methods behave. 

<iframe src="https://qiao.github.io/PathFinding.js/visual/" width="900" height="1200"></iframe>


### A* Implementation

A stand-alone A* planner in python is shown next. Its instructive to go through the code to understand how it works. 

{{< details "A* Planner" "..." >}}

```python
import math
import matplotlib.pyplot as plt

show_animation = True

class AStarPlanner:

    def __init__(self, ox, oy, reso, rr):

        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        open_set, closed_set = dict(), dict()

        # populate the frontier (open set) with the starting node
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.minx),
                         self.calc_grid_position(current.y, self.miny), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)


                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(ngoal, closed_set)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [self.calc_grid_position(ngoal.x, self.minx)], [
            self.calc_grid_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_grid_position(n.x, self.minx))
            ry.append(self.calc_grid_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, minp):
        """
        calc grid position

        :param index:
        :param minp:
        :return:
        """
        pos = index * self.reso + minp
        return pos

    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.reso)

    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.minx)
        py = self.calc_grid_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        # collision check
        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        # map limits
        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx) / self.reso)
        self.ywidth = round((self.maxy - self.miny) / self.reso)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstable positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()

```
{{< /details >}}

Executing the code above results in the animation:

![astar-probabilistic-robotics](images/astar-prob-robotics.gif#center)
*Animation of the A\* algorithm  - from [here](https://github.com/AtsushiSakai/PythonRobotics)*
 
This is [excellent overview](https://arxiv.org/abs/1504.05140) on how the principles of shortest path algorithms are applied in everyday applications such as Google maps directions. Practical implementation considerations are discussed for  multi-modal route finding capabilities where the agent needs to find optimal routes while traversing multiple modes of transportation.

{{< hint info >}}
Although the treatment above is self-contained, if you are missing some algorithmic background, afraid not. There is a free and [excellent book](http://algorithmics.lsi.upc.edu/docs/Dasgupta-Papadimitriou-Vazirani.pdf) to help you with the background behind this chapter. In that book Chapters 3 and 4 are the relevant ones.

{{< /hint >}}