import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

file_path = 'mlruns/0/9181c35090494172bafc17e4ef03ecee/artifacts/episodes/0/ep001131_001178-0-r16-1000.npz'
# file_path = os.path.join('test/episodes', os.listdir('test/episodes')[0])

dream_seq = np.load(file_path)

fig, ax = plt.subplots()


x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
a = dream_seq['image']
ims = []

if len(dream_seq['image'].shape) in (4, 5):
    for seq in dream_seq['image']:
        for img in seq:
            ims.append([ax.imshow(img)])

else:
    for img in dream_seq['image']:
        ims.append([ax.imshow(img)])


ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True,
                                repeat_delay=10)

plt.show()
