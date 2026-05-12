import math

def clamp(value, min, max):
    return max(min(value, max), min)

def rcp(x: float) -> float:
    return 1. / x

def rsqrt(x: float) -> float:
    return 1. / math.sqrt(x)

def sincos(deg: float):
    rad = math.radians(deg)
    return math.sin(rad), math.cos(rad)

def has_unique_characters(string: str) -> bool:
    char_set = set()
    for char in string:
        if char in char_set:
            return False
        char_set.add(char)
    return True

def name_to_idx(char: str) -> int:
    assert len(char) == 1
    if char == 'x':
        return 0
    elif char == 'y':
        return 1
    elif char == 'z':
        return 2
    elif char == 'w':
        return 3
    raise ValueError

def highest_idx(string: str) -> int:
    if 'w' in string: 
        return 3
    elif 'z' in string:
        return 2
    elif 'y' in string:
        return 1
    elif 'x' in string:
        return 0
    raise ValueError