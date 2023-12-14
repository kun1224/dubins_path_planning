from math import pi, tan, sqrt
from itertools import product

from dpp.methods.dubins_path import DubinsPath
from dpp.methods.astar import Astar
from dpp.utils.utils import get_discretized_thetas, round_theta, same_point

from car import SimpleCar


class Node:
    """ Hybrid A* tree node. 
    1. 将连续的状态空间离散化成一个个网格
    2. 每个节点都包含了一个状态，例如小车的位置和朝向。搜索算法会从起点开始，逐步扩展出相邻状态，最终找到一条从起点到终点的路径。
    """

    def __init__(self, grid_pos, pos):

        self.grid_pos = grid_pos    #示节点在网格中的位置，是一个包含了x、y、theta三个值的列表。
        self.pos = pos  #表示节点在实际环境中的位置，是一个包含了x、y、theta三个值的列表。
        self.g = None   #表示从起点到该节点的代价
        self.g_ = None  #表示从起点到该节点的代价，不包括额外的代价
        self.f = None   #表示从起点到该节点的总代价
        self.parent = None  #表示该节点的父节点，是一个Node对象
        self.phi = 0    #表示从父节点到该节点的转角，是一个浮点数。
        self.m = None   #表示从父节点到该节点的运动模式，是一个整数。
        self.branches = []  #表示从父节点到该节点的路径，是一个列表，列表包含了运动模式和所有经过的位置点坐标。

    def __eq__(self, other):#用于判断两个节点是否相等
        # 如果它们在网格中的位置相同，则认为它们相等
        return self.grid_pos == other.grid_pos
    
    def __hash__(self):#用于将节点所在网格转换为一个哈希值

        return hash((self.grid_pos))


class HybridAstar:
    """ Hybrid A* search procedure. """

    def __init__(self, car, grid, reverse, unit_theta=pi/12, dt=1e-2, check_dubins=1):
        
        self.car = car #小车
        self.grid = grid #是否显示网格
        self.reverse = reverse # 是否允许倒车
        self.unit_theta = unit_theta   #角度的离散化单位
        self.dt = dt #每步行驶的时间
        self.check_dubins = check_dubins #检查dubins路径的频率

        self.start = self.car.start_pos #起点位置
        self.goal = self.car.end_pos #终点位置

        self.r = self.car.l / tan(self.car.max_phi) # 转弯半径
        #drive_steps的值表示小车在当前运动模式下需要行驶的步数，以便在对角线方向上穿过一个网格单元格。
        self.drive_steps = int(sqrt(2)*self.grid.cell_size/self.dt) + 1 #是为了确保小车能够到达下一个网格的中心点的行驶步数
        self.arc = self.drive_steps * self.dt #弧长
        self.phil = [-self.car.max_phi, 0, self.car.max_phi] #可能的转角数组
        self.ml = [1, -1] #可能的运动模式

        if reverse:
            self.comb = list(product(self.ml, self.phil)) #可能的运动模式和转角的笛卡尔积组合，用于生成子节点
        else:
            self.comb = list(product([1], self.phil)) #不允许倒车时，只有正向运动

        self.dubins = DubinsPath(self.car) #Dubins路径生成器
        self.astar = Astar(self.grid, self.goal[:2]) #A星算法用于计算启发式函数代价，参数为起点位置，这里将目标点位置作为起点位置
        
        self.w1 = 0.95 # weight for astar heuristic
        self.w2 = 0.05 # weight for simple heuristic
        self.w3 = 0.30 # weight for extra cost of steering angle change
        self.w4 = 0.10 # weight for extra cost of turning
        self.w5 = 2.00 # weight for extra cost of reversing

        self.thetas = get_discretized_thetas(self.unit_theta) #角度的离散化
    
    def construct_node(self, pos):
        """ Create node for a pos. 
        
        """

        theta = pos[2]#角度
        pt = pos[:2]#位置坐标x,y

        theta = round_theta(theta % (2*pi), self.thetas) #将角度转换为离散的角度
        
        cell_id = self.grid.to_cell_id(pt)#将位置转换为网格位置
        grid_pos = cell_id + [theta]#它会根据pos生成一个网格位置grid_pos

        node = Node(grid_pos, pos)#根据实际位置和网格位置生成一个节点

        return node
    
    def simple_heuristic(self, pos):
        """ Heuristic by Manhattan distance. """
        
        return abs(self.goal[0]-pos[0]) + abs(self.goal[1]-pos[1]) #曼哈顿距离的启发式函数
        
    def astar_heuristic(self, pos):
        """ Heuristic by standard astar. """
        #search_path参数为终点位置
        h1 = self.astar.search_path(pos[:2]) * self.grid.cell_size # A*算法的启发式函数：网格数量乘以网格大小
        h2 = self.simple_heuristic(pos[:2]) # Manhattan距离的启发式函数
        
        return self.w1*h1 + self.w2*h2 #两个启发式函数的加权和

    def get_children(self, node,open_, closed_,heu, extra): #node表示当前节点，heu表示要使用的启发式函数，extra表示是否考虑额外的代价
        """ Get successors from a state. 
        通过遍历所有的动作组合，get_children方法可以生成当前节点的所有合法子节点.
        返回 [child子节点, branch到达该子节点的路径点] 列表。
        """

        children = [] #子节点列表
        for m, phi in self.comb: #遍历所有可能的运动模式和转角组合m:运动模式，phi:转角,这里的运动模式是指前进还是倒车，转角是指转向的角度,用于生成子节点

            # don't go back
            if node.m and node.phi == phi and node.m*m == -1: # 如果和上一个节点的转角相同，则不能进行相反的运动模式
                continue

            if node.m and node.m == 1 and m == -1: # 如果当前节点正在前进，那么不能倒车
                continue

            pos = node.pos #当前节点的位置
            branch = [m, pos[:2]]   #包含运动模式m和位置

            #在当前运动模式m下，行驶的步数
            for _ in range(self.drive_steps): #遍历行驶的步数
                pos = self.car.step(pos, phi, m)  #用于模拟小车的运动，返回新的位置
                branch.append(pos[:2]) #将小车在当前运动模式m下经过的每一个位置加入branch

            # check safety of route-----------------------
            pos1 = node.pos if m == 1 else pos #如果是前进，则pos1为当前节点的位置，否则为新的位置
            pos2 = pos if m == 1 else node.pos #如果是前进，则pos2为新的位置，否则为当前节点的位置
            if phi == 0: #如果转角为0，则直接判断直线路径是否安全 
                safe = self.dubins.is_straight_route_safe(pos1, pos2) #判断直线路径是否安全
            else:#如果转角不为0，则判断转弯路径是否安全
                d, c, r = self.car.get_params(pos1, phi) #计算转弯路径的半径、方向和中心点
                safe = self.dubins.is_turning_route_safe(pos1, pos2, d, c, r) #判断转弯路径是否安全
            # --------------------------------------------
            
            if not safe: #如果路径不安全，则跳过该节点
                continue
            
            child = self.construct_node(pos) #路径安全，则生成子节点
            #遍历children,进行剪枝
            for c in children:
                if child.grid_pos[:2] == c[0].grid_pos[:2]: #如果子节点的网格位置和已有的子节点的网格位置相同，则跳过该节点
                    continue
            child.phi = phi #子节点的转角
            child.m = m #子节点的运动模式
            child.parent = node #子节点的父节点是当前节点
            child.g = node.g + self.arc #子节点的代价是当前节点的代价+弧长
            child.g_ = node.g_ + self.arc #子节点的代价是当前节点的代价+弧长

            if extra:
                # extra cost for changing steering angle
                if phi != node.phi: #如果转角和上一个节点的转角不同，则增加额外的代价
                    child.g += self.w3 * self.arc
                
                # extra cost for turning
                if phi != 0: #如果转角不为0，则增加额外的代价
                    child.g += self.w4 * self.arc
                
                # extra cost for reverse
                if m == -1: #如果运动模式为倒车，则增加额外的代价
                    child.g += self.w5 * self.arc

            if heu == 0: 
                child.f = child.g + self.simple_heuristic(child.pos) #计算代价：起点到该节点的代价+simple_heuristic启发式函数的代价组成，这里将当前点作为目标点传入函数
            if heu == 1:
                child.f = child.g + self.astar_heuristic(child.pos) #astar_heuristic启发式函数的代价是从当前节点到目标点的A*算法的代价，这里将当前点作为目标点传入函数
            
            children.append([child, branch]) #
        print("size of children:",len(children))
        return children # 返回子节点列表
    
    def best_final_shot(self, open_, closed_, best, cost, d_route, n=10):   #open_表示待计算的节点列表，closed_表示已计算的节点列表，best表示当前最优节点，cost表示当前最优节点的代价，d_route表示当前最优节点的路径，n表示最多搜索的节点数
        """ 
        Search best final shot(dubins曲线) in open set. 
        """

        open_.sort(key=lambda x: x.f, reverse=False) #根据f值对open_进行排序

        for t in range(min(n, len(open_))): #遍历open_列表中的节点
            best_ = open_[t]    #当前节点
            solutions_ = self.dubins.find_tangents(best_.pos, self.goal)    #计算当前节点到终点的所有dubins曲线路径
            d_route_, cost_, valid_ = self.dubins.best_tangent(solutions_)  #计算最优的dubins曲线路径
        
            if valid_ and cost_ + best_.g_ < cost + best.g_: ## 如果切线路径有效且代价更小，则更新最优路径
                best = best_
                cost = cost_
                d_route = d_route_
        
        if best in open_:   #如果最优节点在open_列表中
            print('Best final shot found!')
            open_.remove(best)  #从open_列表中删除最优节点
            closed_.append(best)    #将最优节点加入closed_列表
        else:
            print('Best final shot not found!')
            
        return best, cost, d_route
    
    def backtracking(self, node):
        """ Backtracking the path. """

        route = []
        while node.parent:
            route.append((node.pos, node.phi, node.m))
            node = node.parent
        
        return list(reversed(route))
    
    def search_path(self, heu=1, extra=False):
        """ Hybrid A* pathfinding. """
        
        root = self.construct_node(self.start)  #生成起点节点
        root.g = float(0)
        root.g_ = float(0)
        
        #根据启发式函数的类型，计算起点节点的代价
        if heu == 0:    
            root.f = root.g + self.simple_heuristic(root.pos)   
        if heu == 1:
            root.f = root.g + self.astar_heuristic(root.pos)

        closed_ = []    #已计算的节点列表
        open_ = [root]  #待计算的节点列表

        count = 0
        while open_:
            count += 1
            best = min(open_, key=lambda x: x.f)    #从open_列表中找到f值最小的节点

            open_.remove(best)  #从open_列表中删除最优节点
            closed_.append(best)    #将最优节点加入closed_列表

            # check dubins path
            if count % self.check_dubins == 0:  #每隔一定的节点数，检查一次dubins路径
                solutions = self.dubins.find_tangents(best.pos, self.goal)  #计算当前best节点到终点的所有dubins曲线路径
                d_route, cost, valid = self.dubins.best_tangent(solutions)  #从所有dubins曲线路径中找到最优的路径
                
                if valid:   
                    best, cost, d_route = self.best_final_shot(open_, closed_, best, cost, d_route) #在open_列表中找到最优的dubins曲线路径
                    route = self.backtracking(best) + d_route   #将最优路径和最优dubins曲线路径组合成最终的路径
                    path = self.car.get_path(self.start, route)   #根据最终路径生成轨迹
                    cost += best.g_ #最终路径的代价是最优dubins曲线路径的代价+最优节点的代价
                    print('Shortest path: {}'.format(round(cost, 2)))
                    print('Total iteration:', count)
                    
                    return path, closed_ #返回最终路径和已计算的节点列表

            children = self.get_children(best, open_,closed_,heu, extra)  #获取最优节点的子节点  

            for child, branch in children:  #遍历子节点

                if child in closed_:    #如果子节点已经在closed_列表中，则跳过该节点
                    continue

                if child not in open_:  #如果子节点不在open_列表中，则将子节点加入open_列表
                    best.branches.append(branch)
                    open_.append(child)

                elif child.g < open_[open_.index(child)].g: #如果子节点在open_列表中，且子节点的代价更小，则更新子节点
                    best.branches.append(branch)
                    print("best.branches:",best.branches)
                    c = open_[open_.index(child)]   #找到open_列表中的子节点
                    p = c.parent    #找到子节点的父节点
                    for b in p.branches:    #遍历父节点的路径
                        if same_point(b[-1], c.pos[:2]):    #如果父节点的路径中的最后一个点和子节点的位置相同
                            p.branches.remove(b)    #则删除该路径   
                            break
                    
                    open_.remove(child) #从open_列表中删除子节点
                    open_.append(child) #将更新后的子节点加入open_列表

        return None, None
