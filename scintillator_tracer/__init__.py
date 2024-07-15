__version__='0.1'

# Some packages used by pvtrace are a little noisy.
import logging
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from . import definitions
from . import geometry_helpers
from . import plotting
from . import processing
from . import worlds
