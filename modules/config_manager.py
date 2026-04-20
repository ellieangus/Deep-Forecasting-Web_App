import json
from typing import Any, Dict


def build_config(**kwargs) -> Dict[str, Any]:
    """Serialise sidebar settings into a JSON-compatible config dict."""
    return dict(kwargs)


def parse_config_bytes(data: bytes) -> Dict[str, Any]:
    """Deserialise a JSON config file uploaded by the user."""
    return json.loads(data)
