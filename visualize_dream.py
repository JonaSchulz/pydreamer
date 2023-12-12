import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

file_path = 'mlruns/0/70983144351f4f4fb98f8c9ba24475b2/artifacts/d2_wm_dream/0013001.npz'
dream_seq = np.load(file_path)

fig, ax = plt.subplots()


x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for seq in dream_seq['image']:
    for img in seq:
        ims.append([ax.imshow(img)])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

plt.show()

