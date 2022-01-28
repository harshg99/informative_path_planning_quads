
import math
import numpy as np
from gym.envs.classic_control import rendering


def circle(x, y, d, box_size, col, res=20):
    box_center = ((x + box_size / 2), (y + box_size / 2))
    thetas = np.linspace(0, 2.0 * math.pi, res)
    circ = rendering.FilledPolygon(list(zip(box_center[0] + np.cos(thetas)*d/ 2, box_center[0] + np.cos(thetas) * d / 2)))
    circ.set_color(col[0], col[1], col[2])
    circ.add_attr(rendering.Transform())
    return circ


def rectangle(x, y, w, h, color, hollow=False):
    ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
    if hollow:
        rect = rendering.PolyLine(ps, True)
    else:
        rect = rendering.FilledPolygon(ps)
    rect.set_color(color[0], color[1], color[2])
    rect.add_attr(rendering.Transform())
    return rect
