"""
=============================================
Generate polygons to fill under 3D line graph
=============================================

Demonstrate how to create polygons which fill the space under a line
graph. In this example polygons are semi-transparent, creating a sort
of 'jagged stained glass' effect.
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

import seaborn as sns
pal=sns.color_palette(palette='magma', n_colors=len(f1M))

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

xs = np.linspace(0, 1, len(f1M[0,:]))
verts = []
zs = np.arange(len(f1M))
for z in zs:
    ys = f1M[z,:]
    #ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

poly = LineCollection(verts,colors=pal)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 1)
ax.set_ylabel('Y')
ax.set_ylim3d(0, len(f1M))
ax.set_zlabel('Z')
ax.set_zlim3d(np.amin(f1M)-1, np.amax(f1M)+1)


plt.show()

#%%
'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = xs
Y = zs
X, Y = np.meshgrid(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, f1M, cmap=cm.plasma,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(np.amin(f1M)-1, np.amax(f1M)+1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.plot_wireframe(X, Y, f1M,color='k',linewidths=0.2)

plt.show()

#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.contourf(X, Y, f1M, cmap=cm.plasma)
