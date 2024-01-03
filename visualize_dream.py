import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

file_path = 'mlruns/0/b0437a39aa3843889c2c520365c3498c/artifacts/episodes/0/ep000000_000000-0-r21-1731.npz'
# file_path = os.path.join('test/episodes', os.listdir('test/episodes')[0])

dream_seq = np.load(file_path)

fig, ax = plt.subplots()


x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
if 'image' not in dream_seq.keys():
    key = 'image_t'

ims = []

if key == 'image_t':
    for img in np.transpose(dream_seq[key], (3, 0, 1, 2)):
        ims.append([ax.imshow(img)])

else:
    if len(dream_seq[key].shape) in (4, 5):
        for seq in dream_seq[key]:
            for img in seq:
                ims.append([ax.imshow(img)])

    else:
        for img in dream_seq[key]:
            ims.append([ax.imshow(img)])


ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                repeat_delay=10)

plt.show()
