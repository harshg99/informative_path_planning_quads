

from gym.envs.classic_control import rendering
import numpy as np

def circle(x, y, d, box_size, col, res=20):
    box_center = ((x + box_size / 2), (y + box_size / 2))
    thetas = np.linspace(0, 2.0 * np.pi, res)
    circ = rendering.FilledPolygon(list(zip(box_center[0] + np.cos(thetas)*d/ 2, box_center[0] + np.cos(thetas) * d / 2)))
    circ.set_color(col[0], col[1], col[2])
    circ.add_attr(rendering.Transform())
    return circ


def rectangle(x, y, w, h, col, hollow=False):
    ps = [(x, y), ((x + w), y), ((x + w), (y + h)), (x, (y + h))]
    if hollow:
        rect = rendering.PolyLine(ps, True)
    else:
        rect = rendering.FilledPolygon(ps)
    rect.set_color(col[0], col[1], col[2])
    rect.add_attr(rendering.Transform())
    return rect
