import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np


def plot(function, min_coords, bound):
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure(figsize=(16, 9))
    fig.set_facecolor((0.95, 0.95, 0.95))
    fig.canvas.set_window_title('Gradient Descent')
    fig.suptitle('Gradient Descent', fontsize=28)
    ax = fig.gca()
    ax.set_axis_bgcolor(fig.get_facecolor())
    ax.tick_params(axis='x', colors=(0.15, 0.15, 0.15))
    ax.tick_params(axis='y', colors=(0.15, 0.15, 0.15))
    x_min, x_max = bound[0], bound[1]
    x = np.linspace(x_min, x_max, 100)

    colors = ['red', 'green', 'yellow', 'cyan']
    i = 0

    n = len(min_coords[0][0])
    if n == 2:
        ax = fig.gca(projection='3d')
        ax.tick_params(axis='z', colors=(0.15, 0.15, 0.15))
        ax.set_axis_bgcolor(fig.get_facecolor())
        y_min, y_max = x_min, x_max
        y = np.linspace(y_min, y_max, 100)
        z = function([x, y], n)
        ax.plot(xs=x, ys=y, zs=z)
        for coords in min_coords:
            xs = coords[0][0]
            ys = coords[0][1]
            zs = function([coords[0][0], coords[0][1]], n)
            label = coords[1]
            ax.scatter(xs=xs, ys=ys, zs=zs, label=label, c=colors[i])
            i += 1
    elif n == 1:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color((0.75, 0.75, 0.75))
        ax.spines['bottom'].set_color((0.75, 0.75, 0.75))
        ax.tick_params(axis='x', colors=(0.15, 0.15, 0.15))
        ax.tick_params(axis='y', colors=(0.15, 0.15, 0.15))
        ax.set_axis_bgcolor(fig.get_facecolor())
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.xlim(bound[0], bound[1])

        x = plt.frange(x_min, x_max, 0.01)
        y = [function(xi, n) for xi in x]
        ax.plot(x, y, alpha=0.85)

        for coords in min_coords:
            ax.scatter(coords[0],
                       function(coords[0], n),
                       label=coords[1],
                       c=colors[i],
                       s=35,
                       edgecolors='black',
                       alpha=0.75)
            i += 1
    else:
        return
    ax.legend()
    plt.show()
