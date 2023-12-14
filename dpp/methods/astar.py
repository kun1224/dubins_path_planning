import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np

from dpp.env.grid import Grid
from dpp.env.environment import Environment
from dpp.test_cases.cases import TestCase
from dpp.utils.utils import distance

from time import time


class Node:
    """ Standard A* node. """

    def __init__(self, cell_id):

        self.cell_id = cell_id
        self.g = None
        self.f = None
        self.parent = None
    
    def __eq__(self, other):

        return self.cell_id == other.cell_id
    
    def __hash__(self):

        return hash((self.cell_id))


class Params:
    """ Store the computed costs. """

    def __init__(self, cell_id, g):

        self.cell_id = cell_id #节点在网格中的位置
        self.g = g #从起点到该节点的代价
    
    def __eq__(self, cell_id):

        return self.cell_id == cell_id
    
    def __hash__(self):

        return hash((self.cell_id))


class Astar:
    """ Standard A*. """

    def __init__(self, grid, start):

        self.grid = grid #是否显示网格
        self.start = self.grid.to_cell_id(start) #起点位置
        self.table = [] #存储起点到终点的代价
    
    def heuristic(self, p1, p2):
        """ Simple Manhattan distance  heuristic. """

        return abs(p2[0]-p1[0]) + abs(p2[1]-p1[1]) #曼哈顿距离
    
    def backtracking(self, node):
        """ Backtracking the path. """

        route = []
        while node.parent:
            route.append((node.cell_id))
            node = node.parent
        
        route.append((self.start))
        
        return list(reversed(route))
    
    def search_path(self, goal): #搜索路径，参数是终点位置
        """ Search the path by astar. """

        goal = self.grid.to_cell_id(goal) #终点位置

        if goal in self.table: #如果终点位置已经计算过了，直接返回代价
            return self.table[self.table.index(goal)].g

        root = Node(self.start) #起点
        root.g = 0 #起点代价
        root.f = root.g + self.heuristic(self.start, goal) #起点代价+启发函数

        closed_ = [] #已经计算过的节点
        open_ = [root] #待计算的节点

        while open_: #如果待计算的节点不为空

            best = min(open_, key=lambda x: x.f) #根据f值选择最优节点，选择代价最小的节点，作为当前节点

            open_.remove(best) #从待计算节点中删除当前节点
            closed_.append(best) #将当前节点加入已计算节点

            if best.cell_id == goal: #如果当前节点是终点
                # route = self.backtracking(best)
                self.table.append(Params(goal, best.g)) #将终点的代价存入table
                return best.g #返回终点的代价

            nbs = self.grid.get_neighbors(best.cell_id)#获取当前节点的邻居节点

            for nb in nbs:
                child = Node(nb)
                child.g = best.g + 1 #邻居节点的代价是当前节点的代价+1
                child.f = child.g + self.heuristic(nb, goal) #邻居节点的f值是邻居节点的代价+启发函数
                child.parent = best #邻居节点的父节点是当前节点

                if child in closed_:
                    continue

                if child not in open_:
                    open_.append(child)
                
                elif child.g < open_[open_.index(child)].g:
                    open_.remove(child)
                    open_.append(child)
        
        return None


def main(grid_on=True):

    tc = TestCase()

    env = Environment(tc.obs)

    grid = Grid(env)

    astar = Astar(grid, tc.start_pos[:2])

    t = time()
    cost, route = astar.search_path(tc.end_pos[:2])
    print(time()-t)
    print('astar cost:', cost)

    pts = []
    for x, y in route:
        x = (x+0.5) * grid.cell_size
        y = (y+0.5) * grid.cell_size
        pts.append([x, y])

    # plot and annimation
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, env.lx)
    ax.set_ylim(0, env.ly)
    ax.set_aspect("equal")

    if grid_on:
        ax.set_xticks(np.arange(0, env.lx, grid.cell_size))
        ax.set_yticks(np.arange(0, env.ly, grid.cell_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        plt.grid()
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    for ob in env.obs:
        ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
    
    ax.plot(tc.start_pos[0], tc.start_pos[1], 'ro', markersize=5)
    ax.plot(tc.end_pos[0], tc.end_pos[1], 'ro', markersize=5)

    lc = LineCollection([pts], linewidth=6)
    ax.add_collection(lc)

    plt.show()


if __name__ == '__main__':
    main()
