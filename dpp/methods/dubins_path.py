from math import tan, atan2, acos, pi
import numpy as np

from dpp.utils.utils import transform, directional_theta, distance


class Params:
    """ Store parameters for different dubins paths. """

    def __init__(self, d):

        self.d = d      # dubins type Dubins路径类型
        self.t1 = None  # first tangent point 第一个切点
        self.t2 = None  # second tangent point 第二个切点
        self.c1 = None  # first center point   第一个圆心
        self.c2 = None  # second center point   第二个圆心
        self.len = None # total travel distance 总行驶距离


class DubinsPath:
    """
    Consider four dubins paths
    - LSL
    - LSR
    - RSL
    - RSR
    and find the shortest obstacle-free one.
    """

    def __init__(self, car):

        self.car = car  #小车对象
        self.r = self.car.l / tan(self.car.max_phi) #转弯半径
        
        # turn left: 1, turn right: -1
        self.direction = {
            'LSL': [1, 1],
            'LSR': [1, -1],
            'RSL': [-1, 1],
            'RSR': [-1, -1]
        }
    
    def find_tangents(self, start_pos, end_pos):
        """ Find the tangents of four dubins paths. """

        self.start_pos = start_pos  #起点位置
        self.end_pos = end_pos  #终点位置

        x1, y1, theta1 = start_pos
        x2, y2, theta2 = end_pos
        
        self.s = np.array(start_pos[:2])    #起点位置
        self.e = np.array(end_pos[:2])  #终点位置
        
        #计算起点和终点的圆心。
        self.lc1 = transform(x1, y1, 0, self.r, theta1, 1)  # 起点左圆心
        # print("self.lc1:",self.lc1)          
        self.rc1 = transform(x1, y1, 0, self.r, theta1, 2)  # 起点右圆心
        # print("self.rc1:",self.rc1)
        self.lc2 = transform(x2, y2, 0, self.r, theta2, 1)  # 终点左圆心
        # print("self.lc2:",self.lc2)
        self.rc2 = transform(x2, y2, 0, self.r, theta2, 2)  # 终点右圆心
        # print("self.rc2:",self.rc2)
        
        solutions = [self._LSL(), self._LSR(), self._RSL(), self._RSR()]   #计算四种Dubins路径的参数 
        solutions = [s for s in solutions if s is not None]   #去除掉无效的Dubins路径
        
        
        return solutions
    
    def get_params(self, dub, c1, c2, t1, t2):  #参数dub表示Dubins路径的类型，c1表示起点圆心，c2表示终点圆心，t1表示起点切点，t2表示终点切点
        """ Calculate the dubins path length. """
        
        v1 = self.s - c1
        v2 = t1     - c1
        v3 = t2     - t1
        v4 = t2     - c2
        v5 = self.e - c2

        delta_theta1 = directional_theta(v1, v2, dub.d[0])  #计算起点到起点切点的航向角之差
        delta_theta2 = directional_theta(v4, v5, dub.d[1])  #计算终点切点到终点的航向角之差

        arc1    = abs(delta_theta1*self.r) #计算弧长
        tangent = np.linalg.norm(v3)    #计算转弯半径切线长度
        arc2    = abs(delta_theta2*self.r)  #计算弧长

        #theta1和theta2表示起点切点和终点切点的航向角，它们的指向相同，相差2*pi的整数倍。
        theta1 = self.start_pos[2] + delta_theta1
        theta2 = self.end_pos[2] - delta_theta2

        dub.t1 = t1.tolist() + [theta1]  #theta表示起点切点的航向角
        # print("dub.t1:",dub.t1)
        dub.t2 = t2.tolist() + [theta2]  #theta表示终点切点的航向角
        # print("dub.t2:",dub.t2)
        dub.c1 = c1 #Dubins路径的起点圆心c1
        dub.c2 = c2 #Dubins路径的终点圆心c2
        dub.len = arc1 + tangent + arc2 #计算路径长度
        
        return dub #返回Dubins路径的参数,包括起点切点，终点切点，起点圆心，终点圆心，路径长度
    
    def _LSL(self):

        lsl = Params(self.direction['LSL']) #创建一个LSL类型的Dubins路径对象

        cline = self.lc2 - self.lc1 #计算起点左圆心和终点左圆心的连线向量
        R = np.linalg.norm(cline) / 2   #计算连线的长度的一半
        theta = atan2(cline[1], cline[0]) - acos(0) #计算连线的角度
                
        t1 = transform(self.lc1[0], self.lc1[1], self.r, 0, theta, 1)   #计算起点左圆心的切点
        t2 = transform(self.lc2[0], self.lc2[1], self.r, 0, theta, 1)   #计算终点左圆心的切点

        lsl = self.get_params(lsl, self.lc1, self.lc2, t1, t2)  #计算LSL类型的Dubins路径的参数

        return lsl  #返回LSL类型的Dubins路径的参数,包括起点切点，终点切点，起点圆心，终点圆心，路径长度

    def _LSR(self):

        lsr = Params(self.direction['LSR'])

        cline = self.rc2 - self.lc1
        R = np.linalg.norm(cline) / 2

        if R < self.r:
            return None
        
        theta = atan2(cline[1], cline[0]) - acos(self.r/R)

        t1 = transform(self.lc1[0], self.lc1[1], self.r, 0, theta, 1)
        t2 = transform(self.rc2[0], self.rc2[1], self.r, 0, theta+pi, 1)

        lsr = self.get_params(lsr, self.lc1, self.rc2, t1, t2)

        return lsr  #返回LSL类型的Dubins路径的参数,包括起点切点，终点切点，起点圆心，终点圆心，路径长度

    def _RSL(self):

        rsl = Params(self.direction['RSL'])

        cline = self.lc2 - self.rc1
        R = np.linalg.norm(cline) / 2

        if R < self.r:
            return None
        
        theta = atan2(cline[1], cline[0]) + acos(self.r/R)

        t1 = transform(self.rc1[0], self.rc1[1], self.r, 0, theta, 1)
        t2 = transform(self.lc2[0], self.lc2[1], self.r, 0, theta+pi, 1)

        rsl = self.get_params(rsl, self.rc1, self.lc2, t1, t2)

        return rsl  #返回LSL类型的Dubins路径的参数,包括起点切点，终点切点，起点圆心，终点圆心，路径长度

    def _RSR(self):

        rsr = Params(self.direction['RSR'])

        cline = self.rc2 - self.rc1
        R = np.linalg.norm(cline) / 2
        theta = atan2(cline[1], cline[0]) + acos(0)

        t1 = transform(self.rc1[0], self.rc1[1], self.r, 0, theta, 1)
        t2 = transform(self.rc2[0], self.rc2[1], self.r, 0, theta, 1)

        rsr = self.get_params(rsr, self.rc1, self.rc2, t1, t2)

        return rsr  #返回LSL类型的Dubins路径的参数,包括起点切点，终点切点，起点圆心，终点圆心，路径长度
    
    def best_tangent(self, solutions):
        """ Get the shortest obstacle-free dubins path. 
        传入的solutions已经按照路径长度从小到大排序了，所以只需要遍历solutions，找到第一个安全的路径即可。
        """
        solutions.sort(key=lambda x: x.len, reverse=False)   #按照路径长度从小到大排序

        pos0 = self.start_pos   #起点位置
        pos1 = self.end_pos #终点位置

        if not solutions:
            return None, None, False
        
        for s in solutions:
            # print("s.len:",s.len)
            route = self.get_route(s)   #获取Dubins曲线的路径点

            safe = self.is_straight_route_safe(s.t1, s.t2)  #判断直线路径是否安全
            if not safe:
                print(safe)
                continue

            safe = self.is_turning_route_safe(pos0, s.t1, s.d[0], s.c1, self.r) #判断第一段转弯路径是否安全
            if not safe:
                print(safe)
                continue

            safe = self.is_turning_route_safe(s.t2, pos1, s.d[1], s.c2, self.r) #判断第二段转弯路径是否安全
            if not safe:
                print(safe)
                continue

            if safe:
                print(safe)
                break
        return route, s.len, safe
    
    def is_straight_route_safe(self, t1, t2):
        """ Check a straight route is safe. """
        # a straight route is simply a rectangle

        vertex1 = self.car.get_car_bounding(t1)
        vertex2 = self.car.get_car_bounding(t2)

        vertex = [vertex2[0], vertex2[1], vertex1[3], vertex1[2]]#将两个矩形组合成一个更大的矩形来判断是否与障碍物相交。

        return self.car.env.rectangle_safe(vertex)
    
    def is_turning_route_safe(self, start_pos, end_pos, d, c, r):
        """ Check if a turning route is safe. """
        # a turning_route is decomposed into:
        #   1. start_pos (checked previously as end_pos)
        #   2. end_pos
        #   3. inner ringsector
        #   4. outer ringsector

        if not self.car.is_pos_safe(end_pos):
            return False
        
        rs_inner, rs_outer = self.construct_ringsectors(start_pos, end_pos, d, c, r)    #构造内外圆环扇形区域
        
        if not self.car.env.ringsector_safe(rs_inner):  #判断内圆环扇形区域是否安全
            return False
        
        if not self.car.env.ringsector_safe(rs_outer):  #判断外圆环扇形区域是否安全
            return False

        return True
    
    def construct_ringsectors(self, start_pos, end_pos, d, c, r):
        """ Construct inner and outer ringsectors of a turning route. """
        
        x, y, theta = start_pos

        delta_theta = end_pos[2] - theta

        p_inner = start_pos[:2]
        id = 1 if d == -1 else 2
        p_outer = transform(x, y, 1.3*self.car.l, 0.4*self.car.l, theta, id)

        r_inner = r - self.car.carw / 2
        r_outer = distance(p_outer, c)

        v_inner = [p_inner[0]-c[0], p_inner[1]-c[1]]
        v_outer = [p_outer[0]-c[0], p_outer[1]-c[1]]

        if d == -1:
            end_inner = atan2(v_inner[1], v_inner[0]) % (2*pi)
            start_inner = (end_inner + delta_theta) % (2*pi)

            end_outer = atan2(v_outer[1], v_outer[0]) % (2*pi)
            start_outer = (end_outer + delta_theta) % (2*pi)
        
        if d == 1:
            start_inner = atan2(v_inner[1], v_inner[0]) % (2*pi)
            end_inner = (start_inner + delta_theta) % (2*pi)

            start_outer = atan2(v_outer[1], v_outer[0]) % (2*pi)
            end_outer = (start_outer + delta_theta) % (2*pi)
        
        rs_inner = [c[0], c[1], r_inner, r, start_inner, end_inner]
        rs_outer = [c[0], c[1], r, r_outer, start_outer, end_outer]

        return rs_inner, rs_outer
    
    def get_route(self, s):
        """ Get the route of dubins path. 
        s的类型为Params，属性包括d,t1,t2,c1,c2,len
        """
        
        phi1 = self.car.max_phi if s.d[0] == 1 else -self.car.max_phi   
        phi2 = self.car.max_phi if s.d[1] == 1 else -self.car.max_phi   

        phil = [phi1, 0, phi2]  #将转弯角度组合成列表
        goal = [s.t1, s.t2, self.end_pos]   #将起点、切点和终点的坐标组成路径点
        ml = [1, 1, 1]  #将每个路径点的运动方式组成列表
        route = list(zip(goal, phil, ml))
        #打印route的内容,格式为:((轨迹点坐标,转弯角度,运动模式),...)
        # for i in range(len(route)):
        #     print("route:",route[i])
        return route
