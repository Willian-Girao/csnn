import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

with open('spiking_digit_sample.npy', 'rb') as f:
    a = np.load(f)

# Create data
x = np.arange(a.shape[1])
y = np.arange(a.shape[2])
X, Y = np.meshgrid(x, y)
frames = range(a.shape[0])  # Values for the z-axis as frames

# Function to update the plot
def update(frame, plot):
    for p in plot:
        p.remove()  # Clear all previous plots
    Z = a[frame]
    
    plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), facecolors=np.where(Z == 1.0, 'k', 'none'))
    plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), color='grey', alpha=0.25)

# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the plot with the initial frame
frame_count = 0
Z = a[frame_count]
plot = [ax.plot_surface(X, Y, np.full_like(X, frame_count), facecolors=np.where(Z == 1.0, 'k', 'none'))]

ax.set_xlabel('x-coor')
ax.set_ylabel('y-coor')
ax.set_zlabel('Time Step')

ax.set_xlim(1, 28)
ax.set_ylim(1, 28)
ax.set_zlim(0, len(frames))

ax.set_xticks(np.arange(1, 29, 1))
ax.set_yticks(np.arange(1, 29, 1))

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])

ani = FuncAnimation(fig, update, frames=frames, fargs=(plot,), interval=1)

ax.grid(False)

writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)

ani.save('spiking_digit_sample.gif', writer=writer)

print('done')