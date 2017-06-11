import skimage.io
import block_match as bm
import scipy.io

n1 = 8    # Block edge
n2 = 32   # Number of similar blocks to look for
ns = 32   # Search neighbourhood edge
nstep = 7 # Reference block grid step

y = skimage.io.imread('cameraman.tif')

# Pick all blocks on a grid.
ref = bm.select_grid_reference(y, n1, ns, nstep)

# Do the actual block matching.
mt, d, df = bm.block_match_self_local(y, ref, n1, n2, ns)

scipy.io.savemat('block_match_results', {'mt':mt, 'ref':ref, 'ns':ns, 'nstep':nstep, 'n1':n1, 'n2':n2})
