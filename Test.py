
import numpy as np
from SOM import SOM
from matplotlib import pyplot as plt

# Training inputs for RGBcolors
colors = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],
     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

# Train a 20x30 SOM with 400 iterations
l = [20,30]
som = SOM(m = l,dim =  3, n_iterations=400)
som.train(colors)

# Get output grid
image_grid = som.get_centroids()
print(image_grid)
image_grid =np.zeros(20*30*3).reshape((20,30,3))
plt.imshow(image_grid)
# Map colours to their closest neurons
mapped = som.map_vects(colors)
print(mapped)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()