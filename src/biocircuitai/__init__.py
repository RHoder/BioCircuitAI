"""BioCircuitAI package root."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("biocircuitai")
except PackageNotFoundError:
    __version__ = "0.1.0"

# convenient re-exports
from .config import CONFIG
