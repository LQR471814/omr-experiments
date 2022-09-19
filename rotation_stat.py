import math
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from utils import clamp, rounded_tuple


def rotate_matrix(matrix, angle: float, callback: Callable[[tuple[float, float], int, int], None]):
    infinity = matrix.shape[1] ** 2

    center = (matrix.shape[0] / 2, matrix.shape[1] / 2)
    slope = clamp(math.tan(angle), ceil=infinity)
    length = 2 * \
        math.sqrt((matrix.shape[0] / 2) ** 2 + (matrix.shape[1] / 2) ** 2)

    def slope_step(m: float):
        return clamp(1 / m if m > 0 else 1, ceil=1)

    def call_point(point: tuple[float, float], i, x):
        point = rounded_tuple(point)
        if point[0] >= matrix.shape[0] or point[0] < 0:
            return
        if point[1] >= matrix.shape[1] or point[1] < 0:
            return
        callback(point, i, x)

    p_slope = -1 / slope if slope > 0 else infinity
    p_step = slope_step(p_slope)

    def trace_perpendicular(point: tuple[float, float], i):
        trace_line(
            length=length,
            center=point,
            slope=p_slope,
            callback=lambda point, x: call_point(point, x, i),
            step=p_step / 2,
        )

    trace_line(
        length=length,
        center=center, slope=slope,
        callback=trace_perpendicular,
        step=slope_step(slope) / 2,
    )


def generate_rotated(angle: float, gap: int = None, size=100, aspect_ratio=1):
    matrix = np.full((size, round(size * aspect_ratio)), 0, dtype=float)

    def set_pixel(point, i, x):
        if gap is None:
            matrix[point[0], point[1]] = x
            return
        if i % gap == 0:
            matrix[point[0], point[1]] = x

    rotate_matrix(matrix, angle, set_pixel)
    return matrix


def trace_line(
    length: float,
    center: tuple[float, float],
    slope: float,
    callback: Callable[[tuple[float, float], int], None],
    step=1,
):
    b = center[1] - slope * center[0]

    dx = length / (2 * math.sqrt(slope**2 + 1))
    start = center[0] - dx

    for i in range(int(dx * 2 / step)):
        x = start + i * step
        y = slope * x + b

        callback((x, y), i)


def patterned_matrix():
    matrix = np.full((100, 100), 0, dtype=float)
    for i in range(100):
        if i % 2 == 0:
            matrix[i, ...] = 1
    return matrix


if __name__ == "__main__":
    matrix = patterned_matrix()

    rotated = generate_rotated(math.pi/6, aspect_ratio=1.5)

    plt.imshow(rotated)
    plt.show()
