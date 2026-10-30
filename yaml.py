"""
YAML wrapper using ruamel.yaml for Python 3.15 compatibility.
"""
from ruamel.yaml import YAML

_yaml = YAML()
_yaml.preserve_quotes = True

def safe_load(stream):
    """Load YAML from stream."""
    return _yaml.load(stream)

def safe_dump(data, stream=None, **kwargs):
    """Dump YAML to stream."""
    if stream is None:
        return _yaml.dump(data)
    _yaml.dump(data, stream)

def load(stream, Loader=None):
    """Load YAML from stream (Loader is ignored)."""
    return _yaml.load(stream)

def dump(data, stream=None, **kwargs):
    """Dump YAML to stream."""
    if stream is None:
        return _yaml.dump(data)
    _yaml.dump(data, stream)

# For compatibility
__all__ = ['safe_load', 'safe_dump', 'load', 'dump', 'YAML']

# Mock the CLoader/CDumper classes
class CLoader:
    pass

class CDumper:
    pass

class SafeLoader:
    pass

class SafeDumper:
    pass
