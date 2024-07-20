"""
Defines the main classes used in managing the photons runs.
"""

import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pvtrace
import typing

from collections import namedtuple
from dataclasses import dataclass

from . import definitions
from . import geometry_helpers
from . import plotting


OUTCOMES = ['absorb', 'exit', 'kill', 'reflect']
RUN_DIR_FMT = 'runs_{time}'
RUN_FILE_FMT = 'run{i}_rays_{x:.3f}x_{y:.3f}y_{z:.3f}z.pkl'


class event_dict(dict):
    @property
    def all(self):
        all = []
        for k in OUTCOMES:
            all += self[k]
        return all


class Manager():


    def __init__(self, world_func: typing.Callable, out_dir: str, make_dirs: bool = True):

        self.time = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.out_dir = os.path.join(out_dir, RUN_DIR_FMT.format(time=self.time))
        self.fig_dir = os.path.join(self.out_dir, 'figures/')
        self.data_dir = os.path.join(self.out_dir, 'data/')
        self.world_func = world_func

        if make_dirs:
            os.makedirs(self.fig_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)


    def load_runs_from_dir(self, pickle_dir: str):
        """
        Loads the Run class pickles from the provided directory.
        """
        
        self.runs = []
        pickles = sorted(f for f in os.listdir(pickle_dir) if f.endswith('.pkl'))
        for pickle_file in sorted(pickles):
            pickle_path = os.path.join(pickle_dir, pickle_file)
            self.runs.append(Run.load(pickle_path))


    def define_grid(
        self,
        vertical_range: tuple,
        horizontal_func,
        steps: int,
        visualize: bool = True
    ):
        """
        Defines the grid points where the light is emitted
        within the crystal.
        """

        self.light_grid_points = geometry_helpers.create_grid_points(
            vertical_range=vertical_range,
            horizontal_func=horizontal_func,
            steps=steps
        )

        if visualize:
            world = self.world_func()['world']
            for i, point in enumerate(self.light_grid_points):
                node = pvtrace.Node(
                    name=f'light node {i}',
                    geometry=pvtrace.Sphere(
                        radius=0.05,
                        material=pvtrace.Material(1)
                    ),
                    parent=world
                )
                node.translate((0, *point))
            scene = pvtrace.Scene(world)
            vis = pvtrace.MeshcatRenderer(wireframe=True, open_browser=True)
            vis.render(scene)
            del(world)
        
        self.plot_grid_points()


    def plot_grid_points(self, ax: plt.Axes = None):
        """
        Plots the grid points defining where the light is emitted
        within the crystal.
        """

        if ax is None:
            fig, ax = plt.subplots(layout='constrained')

        world = self.world_func()['world']
        crystal = get_nodes_from_world(world)['crystal']

        x, y = [], []
        for i, (px, py) in enumerate(self.light_grid_points, start=1):
            x.append(px)
            y.append(py)
            ax.annotate(i, (px-0.02, py+0.05), fontsize=6)

        plotting.draw_node_shape(crystal, ax=ax, null_axis='x', color='black')
        ax.scatter(x, y, color='k')
        # ax.plot( (np.min(x), np.max(x)), (np.max(y)+0.5, np.max(y)+0.5), c='blue')

        ax.set(
            xlabel='[cm]',
            ylabel='[cm]'
        )
        ax.set_aspect('equal', adjustable='box')
        fig_path = os.path.join(self.fig_dir, 'light_grid_map.png')
        plt.savefig(fig_path, dpi=100)

    
    def run(
        self,
        photons_per_grid_point: int,
        seed: int = None,
        visualize: bool = False
    ):
        """
        Run the simulation at each grid point.
        Specifying visualize as True will open a browser window for each run.
        """
        
        width = len(f'{len(self.light_grid_points)}')
        self.runs = []
        for i, point in enumerate(self.light_grid_points, start=1):
            world = self.world_func()['world']
            light = definitions.generate_scintillated_light_node(world)
            light.location = (0, *point)
            scene = pvtrace.Scene(world)
            if visualize:
                all_ray_steps = process_photons_with_visual(
                    scene,
                    photons_per_grid_point,
                    seed,
                    wireframe=True,
                    open_browser=True
                )
            else:
                all_ray_steps = process_photons(scene, photons_per_grid_point, seed)
            events = self.organize_rays(world, all_ray_steps)
            
            index = str(i).zfill(width)
            run = Run((0, *point), events, seed, index)
            run.save(out_dir=self.data_dir)
            self.runs.append(run)

        return self.runs
    

    def organize_rays(
        self,
        world: pvtrace.Node,
        all_ray_steps: list
    ) -> dict:
        return organize_rays(all_ray_steps, get_nodes_from_world(world).values())
    

    def plot_ratio_absorbed(
        self,
        which_node: str,
        ax: plt.Axes = None,
        **pcolorkwargs
    ) -> plt.Axes:
        """
        name specifies the node name, e.g. 'sipm' or 'optical pad'
        """

        return plotting.plot_ratio_absorbed(
            self.runs,
            self.world_func()['world'],
            which_node,
            ax,
            **pcolorkwargs
        )
    

    def plot_all_ratios_absorbed(
        self,
        crystal_kwargs: dict = {},
        optical_pad_kwargs: dict = {},
        sipm_kwargs: dict = {},
    ):
        """
        Plots the ratio of photons absorbed by each object at each grid point.
        """
        
        node_kwargs = {
            'crystal' : {**dict(
                cmap = matplotlib.colormaps['RdYlGn_r'],
                vmin=0.2,
                vmax=0.8
            ), **crystal_kwargs},
            'optical pad' : {**dict(
                cmap = matplotlib.colormaps['RdYlGn_r'],
                vmin=0,
                vmax=0.05
            ), **optical_pad_kwargs},
            'sipm' : {**dict(
                cmap = matplotlib.colormaps['RdYlGn'],
                vmin=0.2,
                vmax=0.5
            ), **sipm_kwargs}
        }

        nodes = get_nodes_from_world(self.world_func()['world'])
        for name, node in nodes.items():
            ax = self.plot_ratio_absorbed(name, **node_kwargs[name])
            
            fig_path = os.path.join(self.fig_dir, f'{name}_absorbed.png')
            print(fig_path)
            plt.savefig(fig_path)


@dataclass
class Run():
    light_location: tuple
    ray_dict: dict
    seed: int
    index: str


    @classmethod
    def load(cls, pickle_path: str):
        """
        Creates an instance of the Run class from the provided pickle.
        """
        
        with open(pickle_path, 'rb') as infile:
            return pickle.load(infile)
    

    def save(self, out_dir: str):
        """
        Save the data to a pickle.
        """

        self.file_path = os.path.join(out_dir, self._format_file_name())
        with open(self.file_path, 'wb') as outfile:
            pickle.dump(self, outfile, 2)

        return self.file_path
    

    def _format_file_name(self) -> str:
        """
        Prepares the file name using the coordinates of the light node.
        """
        
        x, y, z = self.light_location
        file_name = RUN_FILE_FMT.format(i=self.index, x=x, y=y, z=z)

        return file_name


Point = namedtuple('Point', ['x', 'y', 'z'])
def compute_distance(p1: tuple, p2: tuple) -> float:
    return np.sqrt( (p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2 )


def get_nodes_from_world(world: pvtrace.Node) -> dict:
    """
    Returns the desired nodes from the provided world:
    'crystal', 'optical pad' and 'sipm'.
    """

    desired_nodes = ['crystal', 'optical pad', 'sipm']
    nodes = {}
    for child in world.children:
        if child.name in desired_nodes:
            nodes[child.name] = child

    return nodes


def process_photons(scene: pvtrace.Scene, num_photons: int, seed: int) -> list:
    """
    Processes the photons in the scene without a visual.
    Returns a list of each photon's history.
    """
    
    np.random.seed(seed)
    all_ray_steps = []
    for ray in scene.emit(num_photons):
        steps = pvtrace.photon_tracer.follow(scene, ray, maxsteps=1000)
        all_ray_steps.append(steps)
    
    return all_ray_steps


def process_photons_with_visual(
    scene: pvtrace.scene.scene.Scene,
    num_photons: int,
    seed: int,
    **renderer_kwargs
) -> list:
    """
    Processes the photons in the scene and opens a visual.
    Returns a list of each photon's history.
    """
    
    vis = pvtrace.MeshcatRenderer(**renderer_kwargs)
    vis.render(scene)
    np.random.seed(seed)
    all_ray_steps = []
    for ray in scene.emit(num_photons):
        steps = pvtrace.photon_tracer.follow(scene, ray, maxsteps=1000)
        path, decisions = zip(*steps)
        vis.add_ray_path(path)
        all_ray_steps.append(steps)
    
    return all_ray_steps


def organize_rays(
    ray_steps: list,
    nodes: list
) -> dict:
    """
    Sorts the rays based on their outcomes.
    """

    events = event_dict()
    for o in OUTCOMES:
        events[o] = []

    for steps in ray_steps:
        ray, event = steps[-1]
        name = event.name.lower()
        events[name].append(steps)

    for ray_steps in events['absorb']:
        ray, event = ray_steps[-1]
        ray_pos = np.array([ray.position])
        for node in nodes:
            name = (node.name).replace(' ', '_')
            key = f'{name}_absorbed'
            if key not in events:
                events[key] = []

            physical_verts = geometry_helpers.convert_vertices_to_physical(node)
            within = geometry_helpers.in_hull(ray_pos, physical_verts)
            if within:
                events[key].append(ray_steps)
                break # next ray step

    return events


def print_event_report(events: event_dict):

    total = len(events.all)
    print('total number of optical photons:', total)
    for k, v in events.items():
        print(f'\tratio {k}: {len(v)/total}')


def compute_ray_path_lengths(list_of_rays: list) -> list:
    """
    list_of_rays is a list of lists containing the each steps of each ray.
    Returns a list of distances, where each distance is the distance travelled
    by each ray in list_of_rays.
    """

    distance_lists = []
    for steps in list_of_rays:
        ray_distances = []
        previous_step = steps[0]
        for step in steps[1:]:
            p1 = Point(*previous_step[0].position)
            p2 = Point(*step[0].position)
            d = compute_distance(p1, p2)
            previous_step = step
            ray_distances.append(d)
        distance_lists.append(ray_distances)
    distances = [np.sum(d) for d in distance_lists]

    return distances


# def model_func(t, A, K, C):
#     return A * np.exp(K * t) + C


# def fit_exp_linear(t, y, C=0):

#     y = y - C
#     y = np.log(y)
#     K, A_log = np.polyfit(t, y, 1)
#     A = np.exp(A_log)

#     return A, K


# def fit_exp_nonlinear(t, y, p0=None):
#     """
#     from: https://stackoverflow.com/a/3938548
#     """

#     opt_parms, parm_cov = scipy.optimize.curve_fit(
#         model_func,
#         t,
#         y,
#         # maxfev=10000,
#         p0=p0,
#         bounds=(
#             (-np.inf, -np.inf, -10),
#             (np.inf, np.inf, 10)
#         )
#     )
#     A, K, C = opt_parms

#     return A, K, C


def histogram_distances(distances: list, bins: int = 10, ax: plt.Axes = None, fit: bool = False) -> plt.Axes:

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6), layout='constrained')
    
    H, bins = np.histogram(distances, bins=bins)
    _ = ax.stairs(H, bins, color='black', label='simulated')
    ax.set(
        xlabel='Photon path length [cm]',
        ylabel='Counts',
        title=f'mean path length: {np.mean(distances):0.2f} cm',
        ylim=(0.1, ax.get_ylim()[1]),
        yscale='log'
    )

    # if fit:
    #     C0 = -0.1
    #     A, K = fit_exp_linear(bins[:-1], H, C=C0)
    #     # A, K, C0 = fit_exp_nonlinear(bins[:-1], H, p0=(1, 1, -1))
    #     fit_y = model_func(bins[:-1], A, K, C0)
    #     ax.plot(
    #         bins[:-1] + (bins[1]-bins[0])/2,
    #         fit_y,
    #         c='orange',
    #         label=f'exp. fit A = {A:0.1E}, 1/K = {1/K:0.2f}, C = {C0:0.1f}\n A *e^-Kx  + C'
    #     )
    #     ax.legend()

    return ax