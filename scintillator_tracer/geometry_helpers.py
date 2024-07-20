"""
Defines useful functions for manipulating and accessing the geometry in the world.
"""

import math
import numpy as np
import pvtrace
import typing

from scipy.spatial import Delaunay


AXES = {'x': 0, 'y': 1, 'z': 2}


def min_vertex_value(node: pvtrace.Node, which: str) -> float:
    """
    Return the minimum vertex coordinate value along 'which' axis.
    which = 'x', 'y', or 'z'.
    """

    i = AXES[which]
    verts = node.geometry.trimesh.vertices

    return np.min(verts[:,i])


def max_vertex_value(node: pvtrace.Node, which: str) -> float:
    """
    Return the maximum vertex coordinate value along 'which' axis.
    which = 'x', 'y', or 'z'.
    """

    i = AXES[which]
    verts = node.geometry.trimesh.vertices

    return np.max(verts[:,i])


def convert_vertices_to_physical(node: pvtrace.Node) -> np.ndarray:
    """
    Converts the vertex coordinates to physical coordinates using the node's
    pose array. The vertex coordinates are fixed, so the physical (x, y, z)
    coordinate of each point must be obtained using the pose matrix.
    """
    
    verts = node.geometry.trimesh.vertices
    physical = np.array([v@node.pose[:3,:3] for v in verts])
    
    if not np.array_equal(np.eye(3), node.pose[:3,:3]):
        zrot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        physical = physical@zrot

    return physical + node.location


def min_physical_value(node: pvtrace.Node, which: str) -> float:
    """
    Return the minimum physical coordinate value along 'which' axis.
    which = 'x', 'y', or 'z'.

    TODO: Use convert_vertices_to_physical?
    """

    i = AXES[which]
    verts = node.geometry.trimesh.vertices
    verts = np.array([v@node.pose[:3,:3] for v in verts])
    translation = (node.location)[i]

    return np.min(verts[:,i]) + translation


def max_physical_value(node: pvtrace.Node, which: str) -> float:
    """
    Return the maximum physical coordinate value along 'which' axis.
    which = 'x', 'y', or 'z'.

    TODO: Use convert_vertices_to_physical?
    """

    i = AXES[which]
    verts = node.geometry.trimesh.vertices
    verts = np.array([v@node.pose[:3,:3] for v in verts])
    translation = (node.location)[i]

    return np.max(verts[:,i]) + translation


def center_with_origin(node: pvtrace.Node, which: str):
    """
    Center the node on the origin (0, 0, 0).
    """
    
    i = AXES[which]
    c_min, c_max = min_physical_value(node, which), max_physical_value(node, which)
    diff = np.abs(np.abs(c_max) - np.abs(c_min))
    if diff > 0:
        t = [0, 0, 0]
        t[i] = -(c_max + c_min) / 2
        node.translate(t)


def align_bottom_with_xy_plane(node: pvtrace.Node):
    """
    Aligns the bottom-most point of the node with the x-y plane.
    """

    z_min = min_physical_value(node, 'z')
    node.translate((0, 0, -z_min))


def get_node_height(node: pvtrace.Node) -> float:
    """
    Returns the height of the node.
    """

    z_min = min_vertex_value(node, 'z')
    z_max = max_vertex_value(node, 'z')
    
    return z_max - z_min


def stack_nodes(node1: pvtrace.Node, node2: pvtrace.Node):
    """
    Puts node2 on top of node1.
    """
    
    node1_top = max_physical_value(node1, 'z')
    node2_bottom = min_physical_value(node2, 'z')

    diff = node1_top - node2_bottom
    node2.translate((0, 0, diff))


def create_grid_points(vertical_range: tuple, horizontal_func: typing.Callable, steps: int):
    """
    Creates the grid points for the light placement.
    """

    yvals = np.linspace(*vertical_range, steps, endpoint=True)
    pairs = []
    for y in yvals:
        xran = horizontal_func(y)
        xvals = np.linspace(*xran, steps, endpoint=True)
        for x in xvals:
            pairs.append((x, y))
    
    return pairs


def get_unique_coord_pairs(node: pvtrace.Node, null_axis: str) -> list:
    """
    null_axis specifies the axis coming out of the page.
    e.g. 'x'
    """

    i = AXES[null_axis]
    physical_coords = convert_vertices_to_physical(node)
    pairs = np.delete(physical_coords, i, axis=1)

    return np.unique(pairs, axis=0)


def sort_clockwise(coords: list) -> list:
    """
    Sorts the list of coordinates clockwise.

    From: https://stackoverflow.com/a/41856340
    """

    def clockwiseangle_and_distance(point):

        origin = (np.mean(coords[:,0]), np.mean(coords[:,1]))
        refvec = [0, 1]

        vector = [point[0]-origin[0], point[1]-origin[1]] # Between point and origin
        lenvector = math.hypot(vector[0], vector[1]) # ||v||
        if lenvector == 0: # No angle
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to subtract them 
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance should come first.
        return angle, lenvector
    
    return sorted(coords, key=clockwiseangle_and_distance)


def in_hull(p: np.ndarray, hull: np.ndarray) -> bool:
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    Found: https://stackoverflow.com/a/16898636
    """

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
