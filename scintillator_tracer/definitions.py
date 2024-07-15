"""
Defines the materials and sizes for the world building.
"""

import functools
import numpy as np
import pvtrace
import pvtrace.scene.node
import trimesh


UNITS = 'centimeters'


def rotation_matrix_from_vectors(vec1: tuple, vec2: tuple):
    """
    Find the rotation matrix that aligns vec1 to vec2.
    From: https://stackoverflow.com/a/59204638
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


class ESRSurface(pvtrace.FresnelSurfaceDelegate):
    """
    A section of the top surface is covered with a perfect mirrors.
    All other surface obey the Fresnel equations.
    """

    def reflectivity(self, surface, ray, geometry, container, adjacent):
        """
        Return the reflectivity of the part of the surface hit by the ray.
    
        Parameters
        ----------
        surface: Surface
            The surface object belonging to the material.
        ray: Ray
            The ray hitting the surface in the local coordinate system of the `geometry` object.
        geometry: Geometry
            The object being hit (e.g. Sphere, Box, Cylinder, Mesh etc.)
        container: Node
            The node containing the ray.
        adjacent: Node
            The node that will contain the ray if the ray is transmitted.
        """

        # Perfectly reflect
        return 1.0
        
        
    def reflected_direction(self, surface, ray, geometry, container, adjacent):

        # Lambertian scattering at 5%
        # SIGNIFICANTLY slows processing time
        # if np.random.rand() < 0.05:
        #     lamb_dir = np.array(lambertian())
        #     normal = np.array(geometry.normal(ray.position))
        #     rot = rotation_matrix_from_vectors([0, 0, 1], -1*normal)
        #     lamb_dir = rot@lamb_dir
        #     return tuple(lamb_dir)
        # else:
        return super().reflected_direction(surface, ray, geometry, container, adjacent)


class LYSO(pvtrace.Material):
    def __init__(
        self,
        refractive_index: float = 1.8,
        absorption_coefficient: float = 1/16
    ):
        super().__init__(
            refractive_index=refractive_index,
            components=[pvtrace.Absorber(coefficient=absorption_coefficient)]
        )


world_material = pvtrace.Material(refractive_index=1.0)

lyso_material = LYSO() # Default

esr_material = pvtrace.Material(
    refractive_index=1,
    surface=pvtrace.Surface(delegate=ESRSurface())
)

optical_pad_material = pvtrace.Material(
    refractive_index=1.4,
    components=[
        pvtrace.Absorber(coefficient=0.) # Assume perfect emitter
    ]
)

sipm_material = pvtrace.Material(
    refractive_index=1.6,
    components=[
        pvtrace.Absorber(coefficient=100.) # Assume perfect absorber
    ]
)


def load_trimesh(stl_file: str, scaling: float = 1) -> trimesh.base.Trimesh:

    mesh = trimesh.load(stl_file, force='mesh')
    mesh.apply_scale(scaling)

    return mesh


def generate_world(
    name: str = 'world (air)',
    size: tuple = (10, 10, 10),
    material: pvtrace.Material = world_material
):
    return pvtrace.Node(
        name=name,
        geometry=pvtrace.Box(
            size=size,
            material=material
        ),
    )


def generate_crystal(
    stl_file: str,
    scaling: float,
    world: pvtrace.scene.node.Node,
    name: str = 'crystal',
    material: pvtrace.Material = lyso_material
) -> pvtrace.scene.node.Node:
    return pvtrace.Node(
        name=name,
        geometry=pvtrace.Mesh(
            trimesh=load_trimesh(stl_file, scaling),
            material=material
        ),
        parent=world
    )


def generate_esr_shell(
    stl_file: str,
    scaling: float,
    world: pvtrace.scene.node.Node,
    name: str = 'ESR shell',
    material: pvtrace.Material = esr_material
) -> pvtrace.scene.node.Node:
    return pvtrace.Node(
        name=name,
        geometry=pvtrace.Mesh(
            trimesh=load_trimesh(stl_file, scaling),
            material=material
        ),
        parent=world
    )


def generate_optical_pad(
    world: pvtrace.scene.node.Node,
    name: str = 'optical pad',
    size: tuple = (0.4, 0.4, 0.01),
    material: pvtrace.Material = optical_pad_material
) -> pvtrace.scene.node.Node:
    return pvtrace.Node(
        name=name,
        geometry=pvtrace.Box(
            size=size,
            material=material,
        ),
        parent=world
    )


def generate_sipm(
    world,
    name: str = 'sipm',
    size: tuple = (0.6, 0.6, 0.05),
    material: pvtrace.Material = sipm_material
) -> pvtrace.scene.node.Node:
    return pvtrace.Node(
        name=name,
        geometry=pvtrace.Box(
            size=size,
            material=material,
        ),
        parent=world
    )


def generate_scintillated_light_node(
    world: pvtrace.scene.node.Node,
    name: str = 'light',
    wavelength = lambda: 420
) -> pvtrace.scene.node.Node:
    return pvtrace.Node(
        name=name,
        light=pvtrace.Light(
            wavelength=wavelength,
            position=functools.partial(pvtrace.circular_mask, 1e-8),
            direction=pvtrace.isotropic,
            name='scintillated photons'
        ),
        parent=world
    )