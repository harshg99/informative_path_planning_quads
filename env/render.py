

from gym.envs.classic_control import rendering
import numpy as np
import imageio
def circle(x, y, d, box_size, col, res=20):
    center = ((x + box_size / 2), (y + box_size / 2))
    theta = np.linspace(0, 360, res)/360*np.pi*2
    circ = rendering.FilledPolygon(list(zip(center[0] + np.cos(theta)*d/2, center[1] + np.sin(theta) * d/2)))
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

def make_gif(images, fname):
    gif = imageio.mimwrite(fname, images, subrectangles=True)
    print("wrote gif")
    return gif