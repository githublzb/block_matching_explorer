import numpy as np
import skimage as ski

import numpy.random
import scipy.spatial.distance
import skimage.util
import matplotlib.pyplot as plt


def block_match_self(y, ref_coordinates, n1=8, n2=32):
    """
    Block match for self-similarity.
    
    Compute a match table using as search space the same image from where the
    reference block is extracted. The reference block is specified as a tuple of
    coordinates (row,col).
    
    :param y: Input image. Can be grayscale or colour.
    
    :param ref_coordinates: Tuple of reference block coordinates in the form 
    (row, col). row and col must be of the same size, either scalar or some 
    sort of array. If row/col are an array, multiple reference blocks are 
    processed simultaneously.
    
    :param n1: Edge of the patch (first 2 dimensions).
    
    :param n2: Group size. Number of similar patches returned in the match 
    table, per reference block.
    
    :return: Match table. A tuple (row, col). Both row and col are of 
    dimensions (ref_coordinates[0].shape, n2). 
    """

    patch_shape = compute_patch_shape(y, n1)

    # Search is performed in all patches of size patch_shape present in the
    # input image. search_space is a view of the input image y that contains all
    # such patches.
    search_space = skimage.util.view_as_windows(y, patch_shape)

    # Reference patches are all patches requested by the user. Requested
    # coordinates point to the top left corner of the patch in y.
    ref = search_space[ref_coordinates]

    # Flatten arrays for distance computation.
    ref_flat = ref.reshape(-1, np.prod(search_space.shape[y.ndim:]))
    search_space_flat = search_space.reshape(np.prod(search_space.shape[:y.ndim]),
                                             np.prod(search_space.shape[y.ndim:]))

    # Compute pairwise distance between reference blocks and the search space.
    distance = scipy.spatial.distance.cdist(ref_flat, search_space_flat)

    # Sort and truncate.
    sel = np.argpartition(distance, n2)[:, :n2]

    # Return as a tuple (row, col)
    match_table = np.unravel_index(sel, search_space.shape[:2])

    return match_table


def block_match_self_local(y, ref_coordinates, n1=8, n2=8, ns=8,
                         max_memory=2**30):
    """
    Block match for self-similarity, local search.
    
    Compute a match table using as search space a local neighbourhood of the 
    reference block. The reference block is specified as a tuple of    
    coordinates (row,col).
    
    :param y: Input image. Can be grayscale or colour.
    
    :param ref_coordinates: Tuple of reference block coordinates in the form 
    (row, col). row and col must be of the same size, either scalar or some 
    sort of array. If row/col are an array, multiple reference blocks are 
    processed simultaneously.
    
    :param n1: Edge of the patch (first 2 dimensions).
    
    :param n2: Group size. Number of similar patches returned in the match 
    table, per reference block.
    
    :param ns: Size of the local neighbourhood.
    
    :param max_memory: Maximum memory used by this function. Not implemented 
    yet.

    :return: Match table. A tuple (row, col). Both row and col are of 
    dimensions (ref_coordinates[0].size, n2). Note that the reference 
    coordinates are allways flattened and their shape is not reflected in the output.
    """
    patch_shape = compute_patch_shape(y, n1)

    # View image as patches.
    y_patch = ski.util.view_as_windows(y, patch_shape)

    # Reference coordinate arrays are flattened.
    ref_row = np.array(ref_coordinates[0], ndmin=1).reshape(-1, 1)
    ref_col = np.array(ref_coordinates[1], ndmin=1).reshape(-1, 1)

    # Odd ns gives even split between negative and positive offsets. Even ns
    # will give one more candidate block on the positive offsets.
    a = -(ns - 1) // 2 # -floor((ns - 1) / 2)
    b = (ns - 1 + 1) // 2 # ceil((ns - 1) / 2)
    candidate_coordinates = np.mgrid[a:b, a:b]

    # Flatten coordinates and compute absolute.
    candidate_coordinates = candidate_coordinates.reshape(2, 1, -1)
    candidate_row = ref_row + candidate_coordinates[0]
    candidate_col = ref_col + candidate_coordinates[1]

    # But make sure we stay inside the image.
    candidate_row = np.fmax(np.fmin(candidate_row, y_patch.shape[0]), 0)
    candidate_col = np.fmax(np.fmin(candidate_col, y_patch.shape[1]), 0)

    # View of reference blocks and corresponding search space.
    ref = y_patch[(ref_row, ref_col)]
    search_space = y_patch[(candidate_row, candidate_col)]

    # Compute non rooted Euclidean distance.
    difference = search_space - ref
    distance = np.sum(difference ** 2, axis=(2, 3))

    # Find n2 best distances.
    match_idx = np.argpartition(distance, n2)[..., :n2]

    # Generate results.
    tmp = (np.arange(ref_row.size)[:, None], match_idx)
    match_row = candidate_row[tmp]
    match_col = candidate_col[tmp]
    match_distance = distance[tmp]

    full_distance = distance.reshape((-1, ns, ns))

    # Match table is a (row, col) tupple.
    match_table = (match_row, match_col)

    return match_table, match_distance, full_distance


def read_group(y, match_table, n1=8):
    """
    Read a group of patches from image based on match table.
    
    :param y: Image to read patches from.
    
    :param match_table: Match table used to extract patches, generated by 
    block_match_self().
    
    :param n1: Edge of the patch (first 2 dimensions).
    
    :return: Group(s) of patches. View on the input image of shape
    (match_table[0].shape, compute_patch_shape(y, n1)). In practical terms, 
    for grayscale images, the shape is (group_index, patch_index, row, column). 
    """

    patch_shape = compute_patch_shape(y, n1)
    group = skimage.util.view_as_windows(y, patch_shape)[match_table]

    return group


def compute_patch_shape(y, n1):
    """
    Compute shape of patch on input image y.
    
    The shape of a patch is only partially determined by n1, the edge of the 
    patch. For input images of dimension greater than 2, say, colour images, 
    the shape of the patch must include the number of channels. This function
    generates a proper shape tupple taking into account all this.
    
    :param y: Input image which dimensions are to be taken into account.
    
    :param n1: Edge of the patch, for the first 2 dimensions.
    
    :return: Tuple containing patch shape.
    """

    # A patch is n1 by n1 by whatever number of "channels" y has. Usually y
    # has 1 channel, but this should be general.
    if y.ndim > 2:
        patch_shape = (n1, n1, y.shape[2:])
    else:
        patch_shape = (n1, n1)

    return patch_shape


def visualise_match_table(match_table, y, n1=8, ref_coordinates=None,
                          distance=None, distance_full=None):
    """
    Show the match results.
    :param match_table: The return value of any block matching function.
    
    :param y: The image on which to observe the match table.
     
    :param n1: Patch side.
     
    :param ref_coordinates: The same reference coordinates array as provided 
    to the block matching function, in order to identify the reference block.
     
    :param distance: The distance array returned by the block match function,
    in order to visualize the distance between each block and the reference. 
    """

    # 2nd dimension addresses the matches
    group_count, n2 = match_table[0].shape

    # Pick a random group. A group id is a (row, col) tuple of coordinates in
    #  the match table's two first dimensions.
    group_id = np.random.randint(group_count)

    match_entry = (match_table[0][group_id], match_table[1][group_id], )
    group = read_group(y, match_entry, n1)

    if distance is not None:
        distance_entry = distance[group_id]
        axis_count = n2 + 1
    else:
        axis_count = n2

    if ref_coordinates is not None:
        ref_row = np.array(ref_coordinates[0], ndmin=1)[group_id]
        ref_col = np.array(ref_coordinates[1], ndmin=1)[group_id]
    else:
        ref_row = -1
        ref_col = -1

    # Show results.
    axis_count_root = np.ceil(np.sqrt(axis_count))
    plt.figure(figsize=(axis_count_root, axis_count_root))
    plt.set_cmap('gray')
    for r in range(n2):
        b = group[r]
        row = match_entry[0][r]
        col = match_entry[1][r]
        plt.subplot(axis_count_root, axis_count_root, r + 1)
        plt.imshow(b)

        row_rel = row - ref_row
        col_rel = col - ref_col
        if distance is not None:
            d = distance_entry[r]
            tt = plt.title('(%s,%s: %0.6f)' % (row, col, d))
        else:
            tt = plt.title('(%s,%s)' % (match_entry[0][r], match_entry[1][r]))

        if (row_rel, col_rel) == (0, 0):
            tt.set_color('red')

    if distance_full is not None:
        plt.subplot(axis_count_root, axis_count_root, axis_count)
        plt.imshow(distance_full[group_id])
        plt.title('Distance matrix')

    plt.show()

def select_random_ref(y, n1, ns, size):
    """
    Compute the valid range of the reference coordinates.
    :param y: Image to consider as input.
    
    :param n1: 
    :param ns: 
    :param size: Number of coordinates to generate.
    :return: ref_row_lim, ref_col_lim
    """

    a = (ns - 1) // 2 # -floor((ns - 1) / 2)
    b = (ns - 1 + 1) // 2 # ceil((ns - 1) / 2)

    # Make sure reference blocks and search regions are inside the image.
    ref_row_min = a
    ref_row_max = y.shape[0] - b - (n1 - 1)
    ref_row = np.random.randint(ref_row_min, ref_row_max, size=size)

    ref_col_min = a
    ref_col_max = y.shape[1] - b - (n1 - 1)
    ref_col = np.random.randint(ref_col_min, ref_col_max, size=size)

    return ref_row, ref_col


def select_grid_ref(y, n1, ns, nstep, size):
    """
    Compute the valid range of the reference coordinates.
    :param y: Image to consider as input.

    :param n1: 
    :param ns: 
    :param nstep: 
    :param size: Number of coordinates to generate.
    :return: ref_row_lim, ref_col_lim
    """

    a = (ns - 1) // 2  # -floor((ns - 1) / 2)
    b = (ns - 1 + 1) // 2  # ceil((ns - 1) / 2)

    # Make sure reference blocks and search regions are inside the image.
    ref_row_min = a
    ref_row_max = y.shape[0] - b - (n1 - 1)

    ref_col_min = a
    ref_col_max = y.shape[1] - b - (n1 - 1)

    ref = np.mgrid[ref_row_min:ref_row_max:nstep, ref_col_min:ref_col_max:nstep]
    ref = ref.reshape(2, -1)

    return ref[0], ref[1]