import math


def distance(c1: tuple[float, float], c2: tuple[float, float]):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def rounded_tuple(coordinates: tuple[float, float]):
    return (round(coordinates[0]), round(coordinates[1]))


def clamp(n: float, floor=-math.inf, ceil=math.inf):
    if n > ceil:
        return ceil
    if n < floor:
        return floor
    return n
