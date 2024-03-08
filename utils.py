"""
Author: Willian Soares Girão
Contact: wsoaresgirao@gmail.com

Description:    Miscelaneous helper functions (plotting, etc).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_spiking_img(a: np.array):
    
    x = np.arange(a.shape[1])
    y = np.arange(a.shape[2])
    X, Y = np.meshgrid(x, y)
    frames = range(a.shape[0])  # Values for the z-axis as frames

    def update(frame, plot):
        for p in plot:
            p.remove()  # clear previous plots
        Z = a[frame]
        
        plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), facecolors=np.where(Z == 1.0, 'k', 'none'))
        plot[0] = ax.plot_surface(X, Y, np.full_like(X, frame), color='grey', alpha=0.25)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # initial frame
    Z = a[0]
    plot = [ax.plot_surface(X, Y, np.full_like(X, 0), facecolors=np.where(Z == 1.0, 'k', 'none'))]

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

    ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(plot,), interval=1)

    ax.grid(False)

    plt.show()