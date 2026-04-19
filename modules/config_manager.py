import json
from typing import Any, Dict


def build_config(**kwargs) -> Dict[str, Any]:
    return dict(kwargs)


def parse_config_bytes(data: bytes) -> Dict[str, Any]:
    return json.loads(data)
