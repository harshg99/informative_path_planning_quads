# File for testing environment and creating testing environments
from env.searchenv import *
from env.render import *
saveGIF = True


env = SearchEnv(1)
env.createWorld()
action_sequence = np.random.randint(0,4,100)
GIF_frames = []
for j in range(action_sequence.shape[0]):
    env.step(0,ACTIONS[action_sequence[j]])
    GIF_frames.append(env.render(mode='rgb_array'))
    print(env.agents[0].pos)

if saveGIF:
    make_gif(np.array(GIF_frames),
             '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(0,0,0,0))