import numpy as np
import networkx as nx
from time import time
import warnings
warnings.filterwarnings('ignore')

mat = np.random.rand(128, 128)
dir_adj = (mat >= 0.1).astype(np.int8)
np.fill_diagonal(dir_adj, 0)
G_dir = nx.from_numpy_array(dir_adj, create_using=nx.DiGraph())

print("Counting simple cycles for t=0.1...")
t0 = time()
n_cycles = 0
for _ in nx.simple_cycles(G_dir):
    n_cycles += 1
    if n_cycles >= 500: break
print(f"Time for dense graph: {time()-t0:.2f}s")
