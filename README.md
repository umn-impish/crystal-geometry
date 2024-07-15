# scintillator_tracer
This repository contains the models and tests for various scintillator crystal geometries.

The models were created in FreeCad and then exported to STL for use in the `pvtrace` Python package.
Using `pvtrace` we ray trace photons in the scintillator to determine how many reach our detector.
There are STL files for various crystal geometries and accompanying ESR shells.
The ESR shells are constructed such that there is a 1 um gap between the crystal and the ESR, and the ESR is 0.1 mm thick.

See the "notebooks" folder for examples and the methodology used thus far.

The major downside to this is that `pvtrace` is inherently slow due to the ray tracing engine it uses.
It takes several minutes to run a simulation with a few thousand photons.

The world building functions that lay out the models assume that the SiPM readout is at the top of the geometries, where the +z axis is "up."
The STL files are in units of mm while the world building assumes units of centimeters (thus, the STL models are scaled when they're read in).
This inconsistency, while minor, should probably be corrected.

# Assumptions
**BEWARE if you want to make custom models/worlds using the built-in functions and classes!**
- As mentioned above, there's an assumption in where the detector readout is in the geometry.
- The default absorption coefficient for LYSO is 1/16 cm-1. This was informed by research papers; however, this value has large error bars.
- The default optical pad material is assumed to be a perfect transmitter (absorption coefficient = 0), meaning it will not absorb any photons.
- The default SIPM material is assumed to be a perfect absorber. The default absorption coefficient is finite to allow the photons to propagate a small distance into the SiPM volume before being absorbed. **This makes it easier to compute which photons were absorbed by the SiPM since we're a convex hull to determine which volume absorbed each photon**.
- The ESR is assumed to be a perfect reflector.
- The world volume has a refractive index of 1 by default.

# Current geometries
- Rod
- Rod with cut face ("wedge")
- Plate
- Plate with cut face ("chamfered")
- Triangular prism (1 SiPM and 6 SiPM configurations)

# What the simulations DO:
- Allow construction of a detector configuration using various crystal shapes.
- Allow testing light scintillation at various points within the crystal.
- Provide a nice baseline for additional or more complicated crystal shapes.

# What the simulations DON'T DO:
- Lambertian scattering off the ESR surface.
    - This is currently commented out since it greatly increases the run time and sometimes results in crashes due to geometry errors.