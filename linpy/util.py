import math


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min(value, max_val), min_val)

def rcp(x: float) -> float:
    return 1.0 / x

def rsqrt(x: float) -> float:
    return 1.0 / math.sqrt(x)

def sincos(deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return math.sin(rad), math.cos(rad)

def has_unique_characters(string: str) -> bool:
    return len(set(string)) == len(string)

_COMPONENT_MAP: dict[str, int] = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
_VALID_COMPONENTS: frozenset[str] = frozenset(_COMPONENT_MAP)

def name_to_idx(char: str) -> int:
    try:
        return _COMPONENT_MAP[char]
    except KeyError:
        raise ValueError(f"Invalid component name: '{char}'")

def highest_idx(string: str) -> int:
    if 'w' in string:
        return 3
    elif 'z' in string:
        return 2
    elif 'y' in string:
        return 1
    elif 'x' in string:
        return 0
    raise ValueError(f"No valid component in: '{string}'")