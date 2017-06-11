import skimage.io
import block_match as bm
import matplotlib.pyplot as plt

n1 = 8   # Block edge
n2 = 32  # Number of similar blocks to look for
ns = 32  # Search neighbourhood edge

y = skimage.io.imread('0716.png', as_grey=True)

# Pick a random block as reference.
rc = bm.select_random_reference(y, n1, ns, 5)

# Do the actual block matching and build groups.
# mt, d, df = block_match.block_match_self_global(y, rc, n1, n2)
mt, d, df = bm.block_match_self_local(y, rc, n1, n2, ns)

bm.visualise_match_table(mt, y, n1, rc, d, df)

ref, cdt, lab = bm.read_labelled_group(y, mt, rc, n1, ns)

plt.imshow(lab[0].reshape(ns, ns))
plt.show()
