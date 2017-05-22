import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import block_match

n1 = 8   # Block edge
n2 = 32  # Number of similar blocks to look for

y = skimage.io.imread('0716.png', as_grey=True)

# Pick a random block as reference.
ref_row = np.random.randint(y.shape[0] - n1 + 1)
ref_col = np.random.randint(y.shape[1] - n1 + 1)
ref_coordinates = (ref_row, ref_col)

# Do the actual block matching and build groups.
match_table = block_match.block_match_self(y, ref_coordinates, n1, n2)
group = block_match.read_group(y, match_table, n1)

# Show results.
n2root = np.ceil(np.sqrt(n2))
plt.figure(figsize=(n2root, n2root))
for r in range(n2):
    b = group[0, r]
    row = match_table[0][0, r]
    col = match_table[1][0, r]
    plt.subplot(n2root, n2root, r+1)
    plt.imshow(b)

    tt = plt.title('(%s,%s)' % (row, col))
    if row == ref_row and col == ref_col:
        tt.set_color('red')
plt.show()
