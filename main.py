from dataclasses import dataclass
import math
from typing import Any, Union
from PIL import Image, ImageOps # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from rotation_stat import rotate_matrix

from utils import clamp, distance


# * all operations assume grayscale image
def safe_getpixel(image: Image.Image, xy: tuple[float, float], fallback: int = 0):
    if xy[0] < 0 or xy[0] >= image.size[0]:
        return fallback
    if xy[1] < 0 or xy[1] >= image.size[1]:
        return fallback
    return image.getpixel(xy)


def average_contrast(image: Image.Image, convolution: int) -> Image.Image:
    result = Image.new(mode="L", size=image.size)
    i = 0
    for pixel in image.getdata():
        x = i % image.size[0]
        y = i // image.size[0]

        differences = []
        for cy in range(-convolution, convolution + 1):
            for cx in range(-convolution, convolution + 1):
                if cy == 0 and cx == 0:
                    continue
                comparison = safe_getpixel(image, (x + cx, y + cy), 255)
                differences.append(abs(comparison - pixel))

        result.putpixel((x, y), sum(differences) // len(differences))
        i += 1
    return result


def image_field(image: Image.Image, convolution: int) -> tuple[Any, Any, Any, Any]:
    sidelength = max(image.size)

    u = np.full((sidelength, sidelength), 0, dtype=int)
    v = np.full((sidelength, sidelength), 0, dtype=int)

    i = 0
    for pixel in image.getdata():
        x = i % image.size[0]
        y = i // image.size[0]

        dx = 0
        dy = 0

        for cy in range(-convolution, convolution + 1):
            for cx in range(-convolution, convolution + 1):
                dx += cx * \
                    abs((safe_getpixel(image, (x + cx, y + cy)) - pixel) / 255)
                dy += cy * \
                    abs((safe_getpixel(image, (x + cx, y + cy)) - pixel) / 255)

        u[y, x] = dx / convolution ** 2
        v[y, x] = dy / convolution ** 2

        i += 1

    return (
        np.arange(0, sidelength, 1),
        np.arange(0, sidelength, 1),
        u, v
    )


# def filter_skeleton(image: Image.Image, threshold: int):
#     matrix = np.full(image.size, False, dtype=bool)

#     i = 0
#     for pixel in image.getdata():
#         x = i % image.size[0]
#         y = i // image.size[0]

#         if pixel > threshold:
#             matrix[x, y] = True
#         i += 1

#     return matrix

def filter_skeleton(image: Image.Image, threshold: int):
    points: list[tuple[int, int]] = []

    i = 0
    for pixel in image.getdata():
        x = i % image.size[0]
        y = i // image.size[0]

        if pixel > threshold:
            points.append((x, y))
        i += 1

    return points


def show_skeleton(image: Image.Image, points: list[tuple[int, int]]):
    _, ax = plt.subplots(figsize=image.size, layout='constrained')
    ax.scatter([t[0] for t in points], [t[1] for t in points])

    plt.xlim(0, image.size[0])
    plt.ylim(0, image.size[1])
    plt.gca().invert_yaxis()

    plt.show()


Range = tuple[int, int]


@dataclass
class Line:
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    margin: float

    def join(self, other: "Line") -> Union["Line", None]:
        d1 = distance(other.start, self.start)
        d2 = distance(other.end, self.end)

        if d1 > self.thickness + self.margin:
            return None
        if d2 > self.thickness + self.margin:
            return None

        thickness_delta = max(
            clamp(d1 - self.thickness, floor=0),
            clamp(d2 - self.thickness, floor=0),
        )

        return Line(
            start=self.start,
            end=self.end,
            margin=self.margin,
            thickness=self.thickness + thickness_delta,
        )

    def length(self):
        return distance(self.start, self.end)


def stat_rotation(array: np.ndarray, angle: float, target_color=0):
    stats: list[list[Line]] = []

    current_row: int | None = None
    line_start: int | None = None
    last: int | None = None

    def set_stats(point, i, row):
        nonlocal current_row
        nonlocal line_start
        nonlocal last

        if current_row != row:
            stats.append([])
            current_row = row
            return

        value = array[point[0], point[1]]
        is_target = roughly_equal(value, target_color, 20)

        if line_start is None and is_target:
            line_start = row
            last = value
            return

        if line_start is not None and not is_target:
            stats[-1].append(Line())
            line_start = None

    rotate_matrix(array, angle, set_stats)


def roughly_equal(v1: Any, v2: Any, delta: int):
    if abs(int(v2) - int(v1)) < delta:
        return True
    return False


def threshold(array: np.ndarray, threshold=64, target=0):
    matrix = np.full(array.shape, False, dtype=bool)

    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            if roughly_equal(array[y, x], target, threshold):
                matrix[y, x] = True

    return matrix


HORIZONTAL = 0
VERTICAL = 1


def array_to_lines(array: np.ndarray, direction: int):
    stats: list[list[Range]] = []

    y_axis = 0
    x_axis = 1
    if direction == VERTICAL:
        y_axis = 1
        x_axis = 0

    for y in range(array.shape[y_axis]):
        ranges: list[Range] = []

        range_start: int = -1
        last_value: int = 0

        for x in range(array.shape[x_axis]):
            value = array[y, x] if direction == HORIZONTAL else array[x, y]
            if last_value is None:
                last_value = value
                continue
            if last_value != value:
                if value:
                    range_start = x
                elif range_start >= 0:
                    ranges.append((range_start, x))
                    range_start = -1
            last_value = value

        if range_start >= 0:
            ranges.append((range_start, x + 1))

        stats.append(ranges)

    return stats


def filter_lines(
    stats: list[list[Range]],
    joinable_gap=20,
    min_length=100,
):
    filtered: list[list[Range]] = []

    for row in stats:
        lines: list[Range] = []

        for l in row:
            if len(lines) > 0:
                gap_size = l[0] - lines[-1][1]
                # if (gap_size < joinable_gap and
                #         gap_size < max(
                #             l[1] - l[0],
                #             lines[-1][1] - lines[-1][0]
                #         )):
                if gap_size < joinable_gap:
                    lines[-1] = (lines[-1][0], l[1])
                    continue
            lines.append(l)

        filtered.append([l for l in lines if l[1] - l[0] > min_length])

    return filtered


def lines_to_array(stats: list[list[Range]], shape: tuple[int, ...]):
    matrix = np.full(shape, False, dtype=bool)

    for y, row in enumerate(stats):
        for l in row:
            matrix[y, l[0]:l[1]] = True

    return matrix


def line_midpoints(stats: list[list[Range]], shape: tuple[int, ...], vertical=False):
    matrix = np.full(shape, False, dtype=bool)

    for y, row in enumerate(stats):
        for l in row:
            if vertical:
                matrix[math.floor(l[0] + (l[1] - l[0]) / 2), y] = True
                continue
            matrix[y, math.floor(l[0] + (l[1] - l[0]) / 2)] = True

    return matrix


def display_steps(img: Image.Image):
    matrix = np.asarray(img)

    _, ax = plt.subplots(2, 3, figsize=(12, 6), dpi=120)

    ax[0, 0].imshow(matrix)
    ax[0, 0].set_title("original")

    print("calculating lines...", end="", flush=True)
    lines_horizontal = array_to_lines(
        threshold(matrix, threshold=200), direction=HORIZONTAL)
    lines_vertical = array_to_lines(
        threshold(matrix, threshold=200), direction=VERTICAL)
    print("done!")

    ax[1, 0].imshow(lines_to_array(lines_horizontal, matrix.shape))
    ax[1, 0].set_title("lines")

    ax[0, 1].imshow(line_midpoints(lines_horizontal, matrix.shape))
    ax[0, 1].set_title("midpoints horizontal")

    vertical_midpoints = line_midpoints(
        lines_vertical, matrix.shape, vertical=True)
    ax[1, 1].imshow(vertical_midpoints)
    ax[1, 1].set_title("midpoints vertical")

    filtered_vertical = filter_lines(array_to_lines(
        vertical_midpoints, direction=HORIZONTAL
    ))
    ax[0, 2].imshow(lines_to_array(filtered_vertical, matrix.shape))
    ax[0, 2].set_title("filtered vertical")

    plt.show()


def display_final(img: Image.Image):
    matrix = np.asarray(img)

    _, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=120)

    print("calculating lines...", end="", flush=True)
    lines_vertical = array_to_lines(
        threshold(matrix, threshold=200), direction=VERTICAL)
    vertical_midpoints = line_midpoints(
        lines_vertical, matrix.shape, vertical=True)
    filtered_vertical = filter_lines(array_to_lines(
        vertical_midpoints, direction=HORIZONTAL))
    result = lines_to_array(filtered_vertical, matrix.shape)
    print("done!")

    ax[0].imshow(vertical_midpoints)
    ax[1].imshow(result)

    plt.show()


with Image.open("examples/practical/test-1-small.png") as img:
    img = img.convert(mode="L")

    # print("edge detection...", end="", flush=True)
    # result = average_contrast(img, 1)
    # print("done!")

    # field_convolution = 5
    # plot_margin = 1

    display_steps(img)
    # display_final(img)

    # print("calculating field...", end="", flush=True)
    # x, y, u, v = image_field(ImageOps.invert(img), field_convolution)
    # print("done!")
    # plt.quiver(
    #     x, y, u, v,
    #     scale=100 + 60 * field_convolution,
    #     headwidth=2
    # )

    # plt.xlim(-plot_margin, img.size[0] + plot_margin)
    # plt.ylim(-plot_margin, img.size[1] + plot_margin)

    # plt.gca().invert_yaxis()
    # plt.show()

    # print("filtering image...", end="", flush=True)
    # img = ImageOps.invert(img)
    # points = filter_skeleton(img, 64)
    # print("done!")
    # show_skeleton(img, points)

    # result.show()
