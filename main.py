from math import pi
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

from car import SimpleCar
from environment import Environment
from rrt import RRT
from utils.utils import plot_a_car


def main():

    env = Environment()

    start_pos = [6.5, 2, 1.3*pi]
    end_pos = [3, 7.5, -pi/5]
    car = SimpleCar(env, start_pos, end_pos)

    start_state = car.get_car_state(car.start_pos)
    end_state = car.get_car_state(car.end_pos)

    rrt = RRT(car)

    controls = rrt.search_path()
    path = car.get_path(controls)
    xl = [state['pos'][0] for state in path]
    yl = [state['pos'][1] for state in path]

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, env.lx)
    ax.set_ylim(0, env.ly)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for ob in env.obs:
        ax.add_patch(Rectangle((ob.x, ob.y), ob.w, ob.h, fc='gray', ec='k'))
    
    ax = plot_a_car(ax, start_state)
    ax = plot_a_car(ax, end_state)
    ax.plot(xl, yl, color='lime', linewidth=1)

    _path, = ax.plot([], [], color='g', linewidth=3)
    _car = PatchCollection([])
    ax.add_collection(_car)
    frames = len(path) + 1

    def animate(i):

        xl, yl = [], []
        for j in range(min(i+1, len(path))):
            xl.append(path[j]['pos'][0])
            yl.append(path[j]['pos'][1])
        _path.set_data(xl, yl)

        edgecolor = ['k']*5 + ['r']
        facecolor = ['y'] + ['k']*4 + ['r']
        _car.set_paths(path[min(i, len(path)-1)]['model'])
        _car.set_edgecolor(edgecolor)
        _car.set_facecolor(facecolor)

        return _path, _car

    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=1,
                                  repeat=True, blit=True)

    plt.show()


if __name__ == '__main__':
    main()