import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pdb

N=8
vertices=np.zeros((8,2))

vertices[0]=[0,4]
vertices[1]=[3,0]
vertices[2]=[3,5]
vertices[3]=[4,3]
vertices[4]=[6,7]
vertices[5]=[7,5]
vertices[6]=[7,9]
vertices[7]=[9,4]
print("vertices",vertices)
edges=[[0,2],[1,3],[2,3],[2,4],[3,5],[4,5],[4,6],[5,7]]


vertices_min = vertices.min(axis=0)
vertices_max = vertices.max(axis=0)
vertices_normalized = (vertices - vertices_min) / (vertices_max - vertices_min)

# Step 2: Scale and Translate to [-2, -1]
vertices_scaled = vertices_normalized * (1 - (-2)) + (-2)
vertices_scaled = np.around(vertices_scaled, decimals=2)



# Initialize a graph
G = nx.Graph()

# Add vertices as nodes
for i, (x, y) in enumerate(vertices):
    G.add_node(i, pos=(x, y))

# Add edges
G.add_edges_from(edges)

# Get positions
pos = nx.get_node_attributes(G, 'pos')


# Visualize the simplicial complex
fig, ax = plt.subplots(figsize=(8,8))
nx.draw(G, pos=pos, node_size=100, ax=ax)
limits=plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
major_locator=MultipleLocator(1)
ax.yaxis.set_major_locator(major_locator)
ax.xaxis.set_major_locator(major_locator)
ax.grid(linestyle=":",linewidth = 0.5,color = "r",zorder = 0)#pair0
#plt.show()


# Construct the simplicial complex
sc = []
sc.append([[i] for i in range(N)]) # add vertices
sc.append(edges)





simplices_birth=[]
#TODO: add a cone point p
# Compute the birth time for each simplex
for i in range(len(sc)):
	simplices = sc[i]
	for simplex in sc[i]:
		selected_vertices = vertices_scaled[simplex, :]
		v_max = np.max(selected_vertices,axis=0)
		v_min = np.min(selected_vertices,axis=0)
		simplex_birth=[]
		simplex_birth.append(simplex)
		simplex_birth.append(v_max) # R x R
		# TODO: cone the simplex
		simplex_birth.append([v_max[0],-v_min[1]]) # R x Rop
		simplex_birth.append([-v_min[0],v_max[1]]) # Rop x R
		simplex_birth.append([1.,-v_min[1]])  #Rop x Rop
		simplex_birth.append([-v_min[0],1.])  #Rop x Rop
		simplices_birth.append(simplex_birth)

#Then we can compute anything just like an ordinary bifitration


		





