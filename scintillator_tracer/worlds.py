"""
Defines world geometries with the pre-made crystal STL files.
"""

import numpy as np
import os
import pvtrace

from . import definitions
from . import geometry_helpers


STL_DIR = os.path.join(os.path.dirname(__file__), 'stl')

DEFAULT_ROD_STL = os.path.join(STL_DIR, 'rod-Body.stl')
DEFAULT_ROD_ESR_SHELL_STL = os.path.join(STL_DIR, 'rod-esr-shell-Body.stl')

ROD_WEDGE_STL_FMT = os.path.join(STL_DIR, 'rod-{angle}deg-wedge-Body.stl')
ROD_WEDGE_ESR_SHELL_STL_FMT = os.path.join(STL_DIR, 'rod-{angle}deg-wedge-esr-shell-Body.stl')

ISOSCELES_TRIANGULAR_PRISM_STL_FMT = os.path.join(STL_DIR, 'isosceles-{angle}deg-Body.stl')
ISOSCELES_TRIANGULAR_PRISM_ESR_SHELL_STL_FMT = os.path.join(STL_DIR, 'isosceles-{angle}deg-esr-shell-Body.stl')

DEFAULT_TRIANGULAR_PRISM_STL = os.path.join(STL_DIR, 'chamfered-triangular-prism/chamfered-triangular-prism-Body.stl')
DEFAULT_TRIANGULAR_PRISM_ESR_SHELL_STL = os.path.join(STL_DIR, 'chamfered-triangular-prism/chamfered-triangular-prism-esr-shell-Body.stl')

DEFAULT_TRIANGULAR_PRISM_6SIPM_STL = os.path.join(STL_DIR, 'chamfered-triangular-prism/chamfered-triangular-prism-6sipm-Body.stl')
DEFAULT_TRIANGULAR_PRISM_ESR_SHELL_6SIPM_STL = os.path.join(STL_DIR, 'chamfered-triangular-prism/chamfered-triangular-prism-6sipm-esr-shell-Body.stl')

CHAMFERED_PLATE_STL_FMT = os.path.join(STL_DIR, 'chamfered-plate-{angle}deg-Body.stl')
CHAMFERED_PLATE_ESR_SHELL_STL_FMT = os.path.join(STL_DIR, 'chamfered-plate-{angle}deg-esr-shell-Body.stl')


def basic_stack(
    crystal: pvtrace.Node,
    esr_shell: pvtrace.Node,
    optical_pad: pvtrace.Node,
    sipm: pvtrace.Node,
    esr_and_air_thickness: float,
):

    geometry_helpers.align_bottom_with_xy_plane(crystal)
    crystal.translate((0, 0, esr_and_air_thickness)) # Account for ESR thickness and air gap
    geometry_helpers.align_bottom_with_xy_plane(esr_shell)
    for node in [crystal, esr_shell, optical_pad, sipm]:
        geometry_helpers.center_with_origin(node, 'x')
        geometry_helpers.center_with_origin(node, 'y')

    geometry_helpers.stack_nodes(crystal, optical_pad)
    geometry_helpers.stack_nodes(optical_pad, sipm)


def build_rod_world(
    crystal_kwargs: dict = {},
    esr_shell_kwargs: dict = {},
    optical_pad_kwargs: dict = {},
    sipm_kwargs: dict = {},
    world: pvtrace.Node = None
) -> dict:
    """
    The kwargs for each node are the arguments to their respective generation
    functions in the "definitions" submodule. If not specified, the defaults
    will be used.
    """
    
    if world is None:
        world = definitions.generate_world()
    
    DEFAULT_KWARGS = dict(
        crystal = dict(
            stl_file=DEFAULT_ROD_STL,
            scaling=0.1,
            world=world
        ),
        esr_shell = dict(
            stl_file=DEFAULT_ROD_ESR_SHELL_STL,
            scaling=0.1,
            world=world
        ),
        optical_pad = dict(
            world=world
        ),
        sipm = dict(
            world=world
        )
    )
    
    crystal = definitions.generate_crystal(**{**DEFAULT_KWARGS['crystal'], **crystal_kwargs})
    esr_shell = definitions.generate_esr_shell(**{**DEFAULT_KWARGS['esr_shell'], **esr_shell_kwargs})
    optical_pad = definitions.generate_optical_pad(**{**DEFAULT_KWARGS['optical_pad'], **optical_pad_kwargs})
    sipm = definitions.generate_sipm(**{**DEFAULT_KWARGS['sipm'], **sipm_kwargs})

    basic_stack(crystal, esr_shell, optical_pad, sipm, 0.0101)

    nodes = dict(
        world=world,
        crystal=crystal,
        esr_shell=esr_shell,
        optical_pad=optical_pad,
        sipm=sipm
    )

    return nodes


def build_rod_wedge_world(
    angle: float,
    crystal_kwargs: dict = {},
    esr_shell_kwargs: dict = {},
    optical_pad_kwargs: dict = {},
    sipm_kwargs: dict = {},
    world: pvtrace.Node = None
) -> dict:
    """
    Angle is in degrees.
    """

    if world is None:
        world = definitions.generate_world()

    SCALING = np.cos(np.radians(angle))
    DEFAULT_KWARGS = dict(
        crystal = dict(
            stl_file=ROD_WEDGE_STL_FMT.format(angle=angle),
            scaling=0.1,
            world=world
        ),
        esr_shell = dict(
            stl_file=ROD_WEDGE_ESR_SHELL_STL_FMT.format(angle=angle),
            scaling=0.1,
            world=world
        ),
        optical_pad = dict(
            world=world,
            size=(0.4, 0.4/SCALING, 0.01)
        ),
        sipm = dict(
            world=world,
            size=(0.6, 0.6, 0.05)
        )
    )

    crystal = definitions.generate_crystal(**{**DEFAULT_KWARGS['crystal'], **crystal_kwargs})
    esr_shell = definitions.generate_esr_shell(**{**DEFAULT_KWARGS['esr_shell'], **esr_shell_kwargs})
    optical_pad = definitions.generate_optical_pad(**{**DEFAULT_KWARGS['optical_pad'], **optical_pad_kwargs})
    sipm = definitions.generate_sipm(**{**DEFAULT_KWARGS['sipm'], **sipm_kwargs})
    basic_stack(crystal, esr_shell, optical_pad, sipm, 0.0101)
    
    crystal_width = geometry_helpers.max_physical_value(crystal, 'y') - \
        geometry_helpers.min_physical_value(crystal, 'y')
    pad_thickness = geometry_helpers.max_physical_value(optical_pad, 'z') - \
        geometry_helpers.min_physical_value(optical_pad, 'z')
    sipm_thickness = geometry_helpers.max_physical_value(sipm, 'z') - \
        geometry_helpers.min_physical_value(sipm, 'z')

    h = crystal_width / np.tan( np.radians(90-angle) )
    movey = -pad_thickness/2 * np.sin( np.radians(angle) )
    movez = -( h/2 + pad_thickness/2 )

    optical_pad.translate((0, 0, movez))
    optical_pad.rotate(np.radians(angle), (1, 0, 0))
    optical_pad.translate((0, movey, pad_thickness/2 * np.cos(np.radians(angle))))

    movey = -(sipm_thickness/2+pad_thickness) * np.sin(np.radians(angle))
    movez = -(h/2 + pad_thickness + sipm_thickness/2)
    sipm.translate((0, 0, movez))
    sipm.rotate(np.radians(angle), (1, 0, 0))
    sipm.translate((0, movey, (sipm_thickness/2+pad_thickness) * np.cos(np.radians(angle))))

    nodes = dict(
        world=world,
        crystal=crystal,
        esr_shell=esr_shell,
        optical_pad=optical_pad,
        sipm=sipm
    )

    return nodes
    

# def build_triangular_prism_1sipm_world(
#     crystal_material: pvtrace.Material = definitions.lyso_material,
#     esr_shell_material: pvtrace.Material = definitions.esr_material,
#     optical_pad_material: pvtrace.Material = definitions.optical_pad_material,
#     sipm_material: pvtrace.Material = definitions.sipm_material
# ) -> dict:
    
#     world = definitions.generate_world()
#     crystal = definitions.generate_crystal(
#         TRIANGULAR_PRISM_STL,
#         scaling=0.1,
#         world=world,
#         material=crystal_material
#     )
#     esr_shell = definitions.generate_esr_shell(
#         TRIANGULAR_PRISM_ESR_SHELL_STL,
#         scaling=0.1,
#         world=world,
#         material=esr_shell_material
#     )
#     optical_pad = definitions.generate_optical_pad(world, material=optical_pad_material)
#     sipm = definitions.generate_sipm(world, material=sipm_material)

#     basic_stack(crystal, esr_shell, optical_pad, sipm, 0.0101)

#     nodes = dict(
#         world=world,
#         crystal=crystal,
#         esr_shell=esr_shell,
#         optical_pad=optical_pad,
#         sipm=sipm
#     )

#     return nodes
    

# def build_triangular_prism_6sipm_world(
#     crystal_material: pvtrace.Material = definitions.lyso_material,
#     esr_shell_material: pvtrace.Material = definitions.esr_material,
#     optical_pad_material: pvtrace.Material = definitions.optical_pad_material,
#     sipm_material: pvtrace.Material = definitions.sipm_material
# ) -> dict:
    
#     world = definitions.generate_world()
#     crystal = definitions.generate_crystal(
#         TRIANGULAR_PRISM_6SIPM_STL,
#         scaling=0.1,
#         world=world,
#         material=crystal_material
#     )
#     esr_shell = definitions.generate_esr_shell(
#         TRIANGULAR_PRISM_ESR_SHELL_6SIPM_STL,
#         scaling=0.1,
#         world=world,
#         material=esr_shell_material
#     )
#     optical_pad = definitions.generate_optical_pad(
#         world,
#         size=(0.5, 4, 0.01),
#         material=optical_pad_material
#     )
#     sipm = definitions.generate_sipm(world, size=(0.6, 4, 0.1), material=sipm_material)
#     basic_stack(crystal, esr_shell, optical_pad, sipm, 0.0101)

#     nodes = dict(
#         world=world,
#         crystal=crystal,
#         esr_shell=esr_shell,
#         optical_pad=optical_pad,
#         sipm=sipm
#     )

#     return nodes


# def build_isosceles_triangle_world(
#     angle: float,
#     crystal_material: pvtrace.Material = definitions.lyso_material,
#     esr_shell_material: pvtrace.Material = definitions.esr_material,
#     optical_pad_material: pvtrace.Material = definitions.optical_pad_material,
#     sipm_material: pvtrace.Material = definitions.sipm_material
# ) -> dict:
#     """
#     Angle is in degrees.
#     """
    
#     world = definitions.generate_world()
#     crystal = definitions.generate_crystal(
#         ISOSCELES_TRIANGULAR_PRISM_STL_FMT.format(angle=angle),
#         scaling=0.1,
#         world=world,
#         material=crystal_material
#     )
#     esr_shell = definitions.generate_esr_shell(
#         ISOSCELES_TRIANGULAR_PRISM_ESR_SHELL_STL_FMT.format(angle=angle),
#         scaling=0.1,
#         world=world,
#         material=esr_shell_material
#     )
#     optical_pad = definitions.generate_optical_pad(
#         world,
#         size=(0.5, 3.2, 0.01),
#         material=optical_pad_material
#     )
#     sipm = definitions.generate_sipm(world, size=(0.6, 3.6, 0.05), material=sipm_material)

#     h = 0.01 / np.sin(np.radians(angle/2))
#     basic_stack(crystal, esr_shell, optical_pad, sipm, h+0.0001)

#     nodes = dict(
#         world=world,
#         crystal=crystal,
#         esr_shell=esr_shell,
#         optical_pad=optical_pad,
#         sipm=sipm
#     )

#     return nodes


def build_chamfered_plate_world(
    angle: float,
    crystal_kwargs: dict = {},
    esr_shell_kwargs: dict = {},
    optical_pad_kwargs: dict = {},
    sipm_kwargs: dict = {},
    world: pvtrace.Node = None
) -> dict:
    """
    Angle is in degrees.
    """

    if world is None:
        world = definitions.generate_world()

    SCALING = np.cos(np.radians(angle))
    DEFAULT_KWARGS = dict(
        crystal = dict(
            stl_file=CHAMFERED_PLATE_STL_FMT.format(angle=angle),
            scaling=0.1,
            world=world
        ),
        esr_shell = dict(
            stl_file=CHAMFERED_PLATE_ESR_SHELL_STL_FMT.format(angle=angle),
            scaling=0.1,
            world=world
        ),
        optical_pad = dict(
            world=world,
            size=(0.4/SCALING, 4, 0.01)
        ),
        sipm = dict(
            world=world,
            size=(0.6, 4, 0.05)
        )
    )

    crystal = definitions.generate_crystal(**{**DEFAULT_KWARGS['crystal'], **crystal_kwargs})
    esr_shell = definitions.generate_esr_shell(**{**DEFAULT_KWARGS['esr_shell'], **esr_shell_kwargs})
    optical_pad = definitions.generate_optical_pad(**{**DEFAULT_KWARGS['optical_pad'], **optical_pad_kwargs})
    sipm = definitions.generate_sipm(**{**DEFAULT_KWARGS['sipm'], **sipm_kwargs})
    basic_stack(crystal, esr_shell, optical_pad, sipm, 0.0101)
    
    crystal_width = geometry_helpers.max_physical_value(crystal, 'x') - \
        geometry_helpers.min_physical_value(crystal, 'x')
    pad_thickness = geometry_helpers.max_physical_value(optical_pad, 'z') - \
        geometry_helpers.min_physical_value(optical_pad, 'z')
    sipm_thickness = geometry_helpers.max_physical_value(sipm, 'z') - \
        geometry_helpers.min_physical_value(sipm, 'z')

    h = crystal_width / np.tan( np.radians(90-angle) )
    movey = -pad_thickness/2 * np.sin( np.radians(angle) )
    movez = -( h/2 + pad_thickness/2 )

    optical_pad.translate((0, 0, movez))
    optical_pad.rotate(np.radians(angle), (0, 1, 0))
    optical_pad.translate((-movey, 0, pad_thickness/2 * np.cos(np.radians(angle))))

    movey = -(sipm_thickness/2+pad_thickness) * np.sin(np.radians(angle))
    movez = -(h/2 + pad_thickness + sipm_thickness/2)
    sipm.translate((0, 0, movez))
    sipm.rotate(np.radians(angle), (0, 1, 0))
    sipm.translate((-movey, 0, (sipm_thickness/2+pad_thickness) * np.cos(np.radians(angle))))

    nodes = dict(
        world=world,
        crystal=crystal,
        esr_shell=esr_shell,
        optical_pad=optical_pad,
        sipm=sipm
    )

    return nodes
