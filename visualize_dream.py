import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

file_path = 'mlruns/0/f156a68c91404a85b6921c365a9a2301/artifacts/d2_wm_dream/0019001.npz'
# file_path = os.path.join('test/episodes', os.listdir('test/episodes')[0])

dream_seq = np.load(file_path)

fig, ax = plt.subplots()


x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
key = 'image'
if 'image' not in dream_seq.keys():
    key = 'image_t'

ims = []
images = []

if key == 'image_t':
    for img in np.transpose(dream_seq[key], (3, 0, 1, 2)):
        ims.append([ax.imshow(img)])
        images.append(img)

else:
    if len(dream_seq[key].shape) in (4, 5):
        for seq in dream_seq[key]:
            for img in seq:
                ims.append([ax.imshow(img)])
                images.append(img)

    else:
        for img in dream_seq[key]:
            ims.append([ax.imshow(img)])
            images.append(img)


ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                repeat_delay=10)

fig, ax = plt.subplots(1, 4)
for i, _ax in enumerate(ax):
    _ax.imshow(images[20+i])

plt.show()
