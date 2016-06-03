import matplotlib.pylab as plt


def plot(args, title=['']):
    """
    Gets vectors for plot and plots them as I want.
    :param args: Takes 3 vectors:
        vector of points to scatter them on the plot;
        vector of X's as argument;
        vector of Y's as function for X;
    :param title: An optional parameter for naming plots.
    :return: Nothing.
    """
    fig = plt.figure(figsize=(16, 9))
    fig.set_facecolor((0.95, 0.95, 0.95))
    fig.canvas.set_window_title('Regression')
    fig.suptitle('Regression', fontsize=28)
    i = 1
    for points, abscissa, ordinate in args:
        """
        A lot of code just to make it looks better.
        """

        sub = plt.subplot(1, len(args), i)
        sub.spines['top'].set_visible(False)
        sub.spines['right'].set_visible(False)
        sub.spines['left'].set_color((0.75, 0.75, 0.75))
        sub.spines['bottom'].set_color((0.75, 0.75, 0.75))
        sub.tick_params(axis='x', colors=(0.15, 0.15, 0.15))
        sub.tick_params(axis='y', colors=(0.15, 0.15, 0.15))
        sub.set_axis_bgcolor(fig.get_facecolor())
        sub.get_xaxis().tick_bottom()
        sub.get_yaxis().tick_left()
        plt.ylim(min(points[1]) - 1, max(points[1]) + 1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(title[i - 1], fontsize=18)

        """
        """

        plt.scatter(points[0], points[1], c='#3F5D7D', marker='o', edgecolors='black', alpha=0.5)
        plt.plot(abscissa, ordinate, 'r-', alpha=0.85)

        i += 1
    plt.show()
