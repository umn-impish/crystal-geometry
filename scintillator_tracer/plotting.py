"""
Defines commonly use plotting functions for visualizing the ray tracing results.
"""

import itertools
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pvtrace

from . import geometry_helpers
from . import processing


DIR = os.path.dirname(__file__)
STYLE = os.path.join(DIR, 'plot.mplstyle')
plt.style.use(STYLE)


def draw_node_shape(
    node: pvtrace.Node,
    ax: plt.Axes,
    null_axis: str,
    **kwargs
):
    """
    Draws the outline of node on the provided axis.
    null_axis is the axis coming out of the page: e.g. 'x'.
    """
    
    coords = geometry_helpers.get_unique_coord_pairs(node, null_axis)
    coords = np.array(geometry_helpers.sort_clockwise(coords))
    for i in range(len(coords)):
        pair1 = coords[i,:]
        pair2 = coords[(i+1)%len(coords),:]
        x = (pair1[0], pair2[0])
        y = (pair1[1], pair2[1])
        ax.plot(x, y, **kwargs)
    # ax.scatter(coords[:,0], coords[:,1], s=4, **kwargs)


def prep_ratio_absorbed(runs: list, key: str, shape: tuple) -> tuple:
    """
    Organizes the results from the list of runs into a surface to be plotted
    with pcolormesh. key specifies which object to plot to results of.
    """

    surface = []
    x, y = [], []
    for run in runs:
        x.append(run.light_location[1])
        y.append(run.light_location[2])
        ratio = len(run.ray_dict[key]) / len(run.ray_dict.all)
        surface.append(ratio)

    x = np.array(x).reshape(shape)
    y = np.array(y).reshape(shape)
    surface = np.array(surface).reshape(shape)

    return x, y, surface


def plot_ratio_absorbed(
    runs: list,
    world: pvtrace.Node,
    which_node: str,
    ax: plt.Axes = None,
    show_colorbar: bool = True,
    show_legend: bool = True,
    **pcolorkwargs
):
    """
    Plots the ratio absorbed by which_node.
    "runs" is a list of "Run" class objects (in processing.py).
    """

    default_kwargs = dict(cmap=matplotlib.colormaps['RdYlGn'])
    pcolorkwargs = {**default_kwargs, **pcolorkwargs}

    nodes = processing.get_nodes_from_world(world)
    steps = int(np.sqrt(len(runs)))

    key = f'{which_node.replace(" ", "_")}_absorbed'
    Y, Z, surface = prep_ratio_absorbed(runs, key, (steps, steps))
    
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
        
    C = ax.pcolormesh(Y, Z, surface, **pcolorkwargs)
    ax.scatter(Y.flatten(), Z.flatten(), color='black', marker='.', s=2)

    if show_colorbar:
        plt.colorbar(C, label=f'ratio absorbed by {which_node}')

    colors = itertools.cycle(plt.cm.plasma(np.linspace(0, 1, len(nodes)+1)))
    
    patches = []
    for node in nodes.values():
        color = next(colors)
        draw_node_shape(node, ax, 'x', color=color)
        patches.append(mpatches.Patch(color=color, label=(node.name).replace('_', ' ')))
    if show_legend:
        ax.legend(handles=patches)

    # Make square.
    xdiff = np.diff(ax.get_xlim())[0]
    ydiff = np.diff(ax.get_ylim())[0]
    if xdiff > ydiff:
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0]-(xdiff-ydiff)/2, ylim[1]+(xdiff-ydiff)/2)
    elif xdiff < ydiff:
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0]-(ydiff-xdiff)/2, xlim[1]+(ydiff-xdiff)/2)

    ax.set(
        xlabel='Horizontal coordinate [cm]',
        ylabel='Vertical coordinate [cm]',
        title=f'photons absorbed by {which_node}\nmean absorption ratio: {np.mean(surface):0.3f}'
    )

    return ax