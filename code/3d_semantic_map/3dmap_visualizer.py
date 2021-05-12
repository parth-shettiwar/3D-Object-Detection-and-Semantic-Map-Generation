from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import cm

cmap = cm.plasma

positions = []
sizes = []
with open('../../results/votenet-ensemble_semantic-map.json') as f:
  data = json.load(f)
for i in range(len(data["results"]['objects'])):
  positions.append(tuple(data["results"]['objects'][i]['centroid']))
  sizes.append(tuple(data["results"]['objects'][i]['extent']))
def cuboid_data2(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt2(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data2(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6), **kwargs)
    

# positions = [(-3,5,-2),(1,7,1)]
# sizes = [(4,5,3), (3,3,7)]
colors = ["crimson","limegreen","red","blue"]
colours_main = []
for i in range(len(data["results"]['objects'])):
  colours_main.append(colors[i%4])
# quantity = [0.1, 0.11, 5, 10]
# colors = cmap(quantity)

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.set_aspect('equal')

pc = plotCubeAt2(positions,sizes,colors=colours_main, edgecolor="k")
ax.add_collection3d(pc)    

def r1(tuple_1):
    return tuple_1[0]
def r2(tuple_1):
    return tuple_1[1]
def r3(tuple_1):
    return tuple_1[2]        


print(len(positions))
print(positions)
x_min = min(positions,key=r1)[0]-2
y_min = min(positions,key=r2)[1]-2
z_min = min(positions,key=r3)[2]-2
x_max = max(positions,key=r1)[0]+1
y_max = max(positions,key=r2)[1]+1
z_max = max(positions,key=r3)[2]+1


ax.set_xlim([x_min,x_max])
ax.set_ylim([y_min,y_max])
ax.set_zlim([0,z_max])

print(x_max,x_min,y_max,y_min,z_max,z_min)

# ax.set_xlim([-4,6])
# ax.set_ylim([-1,13])
# ax.set_zlim([-1,7])

plt.show()