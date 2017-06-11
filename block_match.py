import numpy as np
import skimage as ski

import numpy.random
import scipy.spatial.distance
import skimage.util
import matplotlib.pyplot as plt


def block_match_self(y, reference_x, n1, n2):
    """
    Block match for self-similarity.
    
    Compute a match table using as search space the same image from where the
    reference block is extracted. The reference block is specified as a tuple of
    coordinates (row,col).
    
    :param y: Input image. Can be grayscale or colour.
    
    :param reference_x: Tuple of reference block coordinates in the form
    (row, col). row and col must be of the same size, either scalar or some 
    sort of array. If row/col are an array, multiple reference blocks are 
    processed simultaneously.
    
    :param n1: Edge of the patch (first 2 dimensions).
    
    :param n2: Group size. Number of similar patches returned in the match 
    table, per reference block.
    
    :return: Match table. A tuple (row, col). Both row and col are of 
    dimensions (reference_x[0].size, n2).
    """

    patch_shape = compute_patch_shape(y, n1)

    # Search is performed in all patches of size patch_shape present in the
    # input image. search_space is a view of the input image y that contains all
    # such patches.
    candidate = skimage.util.view_as_windows(y, patch_shape)

    # Reference patches are all patches requested by the user. Requested
    # coordinates point to the top left corner of the patch in y.
    reference = candidate[reference_x]

    # Flatten arrays for distance computation.
    reference = reference.reshape(-1, np.prod(candidate.shape[y.ndim:]))
    candidate = candidate.reshape(np.prod(candidate.shape[:y.ndim]),
                                  np.prod(candidate.shape[y.ndim:]))

    # Compute pairwise distance between reference blocks and the candidates.
    distance = scipy.spatial.distance.cdist(reference, candidate)

    # Sort and truncate.
    match_idx = np.argpartition(distance, n2)[:, :n2]

    # Return as a tuple (row, col)
    match_table = np.unravel_index(match_idx, candidate.shape[:2])

    return match_table


def block_match_self_local(y, reference_x, n1, n2, ns, max_memory=2**30):
    """
    Block match for self-similarity, local search.

    Compute a match table using as search space a local neighbourhood of the
    reference block. The reference block is specified as a tuple of
    coordinates (row,col).

    :param y: Input image. Can be grayscale or colour.

    :param reference_x: Tuple of reference block coordinates in the form
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
    dimensions (reference_x[0].size, n2). Note that the reference coordinates
    are always flattened and their shape is not reflected in the output.
    """
    patch_shape = compute_patch_shape(y, n1)

    # Patch view of the input image.
    y_patch = ski.util.view_as_windows(y, patch_shape)

    # Enforce bounds and shape on reference coordinate vector.
    reference_x = flatten_reference_coordinates(reference_x)
    reference_x = sanitize_coordinates(reference_x, y_patch.shape)

    candidate_x = compute_candidate_coordinates(reference_x, y_patch.shape, ns)

    # View of reference and corresponding candidate patches.
    reference = y_patch[reference_x]
    candidate = y_patch[candidate_x]

    # Compute non rooted Euclidean distance.
    #
    # Computing the distance matrix corresponding to a single reference block
    # requires (ns**2 * n1**2 * 2**3) bytes of space, which is the size of
    # the difference matrix considering numbers stored as doubles. In words,
    # there are ns**2 candidate blocks, each of size n1**2, each element
    # taking 2**3 bytes. The size of the distance matrix pales in comparison,
    # a mere ns**2 * 2**3. As a typical example, n1 = 8, ns = 32, the size of
    # the difference matrix is 2**(5*2 + 3*2 + 3) = 2**19 bytes or 512KB. The
    # size of the distance matrix is 2**(5*2 + 3) = 2**13 bytes or 8KB. When
    # limiting the memory usage I am assuming that all the consumed memory is
    # due to the difference matrix.

    distance = np.empty((reference_x[0].size, ns**2, ))
    # Assumes storage as float64 (double)
    bytes_per_reference = ns**2 * n1**2 * 2**3
    reference_per_cycle = max_memory // bytes_per_reference
    for r in range(0, reference_x[0].size, reference_per_cycle):
        low = r
        high = min(r + reference_per_cycle, reference_x[0].size)
        reference_slice = reference[low:high, :]
        candidate_slice = candidate[low:high, :]

        # Compute non rooted Euclidean distance.
        difference = candidate_slice - reference_slice
        distance[low:high, :] = np.sum(difference ** 2, axis=(2, 3))

    # Find n2 best distances.
    match_idx = np.argpartition(distance, n2)[..., :n2]

    # Generate results.
    tmp = (np.arange(reference_x[0].size)[:, None], match_idx, )
    match_row = candidate_x[0][tmp]
    match_col = candidate_x[1][tmp]
    match_distance = distance[tmp]

    full_distance = distance.reshape((-1, ns, ns))

    # Match table is a (row, col) tuple.
    match_table = (match_row, match_col)

    return match_table, match_distance, full_distance


def read_labelled_group(y, match_table, reference_x, n1, ns):

    # View image as patches.
    patch_shape = compute_patch_shape(y, n1)
    y_patch = ski.util.view_as_windows(y, patch_shape)
    y_patch = y_patch.reshape(y_patch.shape[:2] + (-1, ))

    reference_x = flatten_reference_coordinates(reference_x)
    reference_x = sanitize_coordinates(reference_x, y_patch.shape)

    candidate_x = compute_candidate_coordinates(reference_x, y_patch.shape, ns)

    reference = y_patch[reference_x]
    candidate = y_patch[candidate_x]

    # Label is true when block is similar.
    candidate_row, candidate_col = candidate_x
    match_row, match_col = match_table
    label_row = (candidate_row[:, :, None] == match_row[:, None, :])
    label_col = (candidate_col[:, :, None] == match_col[:, None, :])
    label = np.logical_and(label_row, label_col)
    label = np.any(label, axis=-1)

    return reference, candidate, label


def compute_candidate_coordinates(reference_x, patch_shape, ns):
    """ Compute coordinates of candidate patches. """

    # Odd ns gives even split between negative and positive offsets. Even ns
    # will give one more candidate block on the positive offsets.
    a = -(ns - 1) // 2  # -floor((ns - 1) / 2)
    b = (ns - 1 + 1) // 2  # ceil((ns - 1) / 2)
    candidate_x = np.mgrid[a:b, a:b].reshape(2, 1, -1)

    # Compute absolute coordinates.
    candidate_row = reference_x[0] + candidate_x[0]
    candidate_col = reference_x[1] + candidate_x[1]

    # We always work with tuples.
    candidate_x = (candidate_row, candidate_col, )
    candidate_x = sanitize_coordinates(candidate_x, patch_shape)

    return candidate_x


def flatten_reference_coordinates(x):
    """ Turn a N dimensional coordinate array into appropriate size. """
    x_flat = x[0].reshape(-1, 1), x[1].reshape(-1, 1),

    return x_flat


def sanitize_coordinates(x, s):
    """ Make sure coordinates specify blocks that are fully inside the image."""

    row = np.fmax(np.fmin(x[0], s[0]), 0)
    col = np.fmax(np.fmin(x[1], s[1]), 0)

    return row, col


def read_group(y, match_table, n1):
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


def visualise_match_table(match_table, y, n1, reference_x=None,
                          distance=None, distance_full=None):
    """
    Show the match results.
    :param match_table: The return value of any block matching function.

    :param y: The image on which to observe the match table.

    :param n1: Patch side.

    :param reference_x: The same reference coordinates array as provided
    to the block matching function, in order to identify the reference block.

    :param distance: The distance array returned by the block match function.
    If present, each individual plot title will contain the distance between
    that block and the reference.

    :param distance_full: The other distance array returned by the block
    match function. This one contains the distances relative to the whole
    search region of each reference block. If present, an extra plot will be
    made showing the distances as a surface.
    """

    # 2nd dimension addresses the matches
    group_count, n2 = match_table[0].shape

    # Pick a random group.
    group_id = np.random.randint(group_count)

    match_entry = (match_table[0][group_id], match_table[1][group_id], )
    group = read_group(y, match_entry, n1)

    if distance is not None:
        distance_entry = distance[group_id]
        axis_count = n2 + 1
    else:
        axis_count = n2

    if reference_x is not None:
        reference_row = np.array(reference_x[0], ndmin=1)[group_id]
        reference_col = np.array(reference_x[1], ndmin=1)[group_id]
    else:
        reference_row = -1
        reference_col = -1

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

        row_rel = row - reference_row
        col_rel = col - reference_col
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


def select_random_reference(y, n1, ns, size=1):
    """
    Compute the valid range of the reference coordinates.
    :param y: Image to consider as input.
    
    :param n1: 
    :param ns: 
    :param size: Number of coordinates to generate.
    :return: reference_x: Coordinate tuple that can be used with block
    matching functions.
    """

    a = (ns - 1) // 2  # -floor((ns - 1) / 2)
    b = (ns - 1 + 1) // 2  # ceil((ns - 1) / 2)

    # Make sure reference blocks and search regions are inside the image.
    ref_row_min = a
    ref_row_max = y.shape[0] - b - (n1 - 1)
    ref_row = np.random.randint(ref_row_min, ref_row_max, size=size)

    ref_col_min = a
    ref_col_max = y.shape[1] - b - (n1 - 1)
    ref_col = np.random.randint(ref_col_min, ref_col_max, size=size)

    reference_x = ref_row, ref_col

    return reference_x


def select_grid_reference(y, n1, ns, nstep):
    """
    Compute the valid range of the reference coordinates.
    :param y: Image to consider as input. Only the size matters.

    :param n1: 
    :param ns: 
    :param nstep:
    :return: reference_x: Coordinate tuple that can be used with block
    matching functions.
    """

    a = (ns - 1) // 2  # -floor((ns - 1) / 2)
    b = (ns - 1 + 1) // 2  # ceil((ns - 1) / 2)

    # Make sure reference blocks and search regions are inside the image.
    reference_row_min = a
    reference_row_max = y.shape[0] - b - (n1 - 1)

    reference_col_min = a
    reference_col_max = y.shape[1] - b - (n1 - 1)

    ref = np.mgrid[reference_row_min:reference_row_max:nstep,
                   reference_col_min:reference_col_max:nstep]
    ref = ref.reshape(2, -1)

    reference_x = ref[0], ref[1]

    return reference_x
