import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pdb

'''
N=8
vertices=np.zeros((N,2))
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
'''

N=5
vertices=np.zeros((N,2))
vertices[0]=[0,1]
vertices[1]=[1,1]
vertices[2]=[2,2]
vertices[3]=[2,0]
vertices[4]=[3,1]
edges=[[0,1],[1,2],[1,3],[2,4],[3,4],[2,3]]
faces=[[1,2,3]]



vertices_min = vertices.min(axis=0)
vertices_max = vertices.max(axis=0)

vertices_normalized = (vertices - vertices_min) / (vertices_max - vertices_min)
# Step 2: Scale and Translate to [-2, -1]
vertices_scaled = vertices_normalized - 2 
vertices_scaled = np.around(vertices_scaled, decimals=1)

print(vertices_scaled)

# Initialize a graph
G = nx.Graph()

# Add vertices as nodes
for i, (x, y) in enumerate(vertices_scaled):
    G.add_node(i, pos=(x, y))

# Add edges
G.add_edges_from(edges)

# Get positions
pos = nx.get_node_attributes(G, 'pos')


# Visualize the simplicial complex
fig, ax = plt.subplots(figsize=(8,8))
nx.draw(G, pos=pos, node_size=100, ax=ax)

for face in faces:
	(x0, y0) = vertices_scaled[face[0]]
	(x1, y1) = vertices_scaled[face[1]]
	(x2, y2) = vertices_scaled[face[2]]
	tri = plt.Polygon([[ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],edgecolor = 'black', facecolor = plt.cm.Blues(0.6),zorder = 2, alpha=0.4, lw=0.5)
	ax.add_patch(tri)


limits=plt.axis('on')
#ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#major_locator=MultipleLocator(1)
#ax.yaxis.set_major_locator(major_locator)
#ax.xaxis.set_major_locator(major_locator)
#ax.grid(linestyle=":",linewidth = 0.5,color = "r",zorder = 0)#pair0

arrowprops={'arrowstyle': '-', 'ls':'--'}
for i in range(len(vertices_scaled)):
	vertex = vertices_scaled[i]
	x = vertex[0]
	y = vertex[1]
	plt.annotate(str(x), xy=(x,y), xytext=(x, 0), textcoords=plt.gca().get_xaxis_transform(),arrowprops=arrowprops,va='top', ha='center')
	plt.annotate(str(y), xy=(x,y), xytext=(0, y), textcoords=plt.gca().get_yaxis_transform(),arrowprops=arrowprops,va='center', ha='right')
	ax.annotate(i, (x+0.02, y+0.02))
plt.show()


# Construct the simplicial complex
sc = []
sc.append([[i] for i in range(N)]) # add vertices
sc.append(edges)
sc.append(faces)





simplices_birth=[]
temp_simp=[]
simplex_birth=[[N],[-3,-3]]
simplices_birth.append(simplex_birth)
# Compute the birth time for each simplex
for i in range(len(sc)):
	simplices = sc[i]
	for simplex in simplices:
		selected_vertices = vertices_scaled[simplex, :]
		v_max = np.max(selected_vertices,axis=0).tolist()
		v_min = np.min(selected_vertices,axis=0).tolist()
		simplex_birth=[]
		simplex_birth.append(simplex)
		simplex_birth.append(v_max) # R x R
		simplices_birth.append(simplex_birth)
		simplex_birth2=[]
		temp_simp = simplex.copy()
		temp_simp.append(N)
		simplex_birth2.append(temp_simp)
		simplex_birth2.append([v_max[0],-v_min[1]]) # R x Rop
		simplex_birth2.append([-v_min[0],v_max[1]]) # Rop x R
		simplices_birth.append(simplex_birth2)

#print(simplices_birth)
# Write a txt file for RIVET
# Initialize the string to write to the file
output_str = "--datatype bifiltration\n--xlabel x values\n--ylabel y values\n\n#data\n"

for simplex in simplices_birth:
    # Extract the indices and coordinates
    indices, *coords = simplex
    # Format the indices part
    indices_str = ' '.join(map(str, indices))
    # Flatten and format the coordinates part
    coords_str = ' '.join(f"{coord:.1f}" for sublist in coords for coord in sublist)
    # Append formatted string to output
    output_str += f"{indices_str} ; {coords_str}\n"

# Write the string to a file
with open("output.txt", "w") as file:
    file.write(output_str)


#Then we can compute anything just like an ordinary bifitration

		





