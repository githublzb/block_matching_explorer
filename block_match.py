import numpy as np
import skimage.util

def block_match(ref, y, n2, nstep=1):
    # Only grayscale or color images.
    if ref.ndim > 3:
        raise TypeError('Only grayscale (2 channels) or colour (3 channels) '
                        'inputs allowed.')
    # ref an y must be compatible
    if y.ndim != ref.ndim:
        raise TypeError('ref and y must have the same number of dimensions.')

    if ref.ndim == 2:
        ref = ref[:, :, None]
        y = y[:, :, None]

    if y.shape[2] != ref.shape[2]:
        raise TypeError('ref and y must have the same number of '
                        'colour channels.')

    step = nstep, nstep, 1
    blk = skimage.util.view_as_windows(y, ref.shape, step)

    # Simplified l2 (no need for square root)
    dif = ref - blk
    dif = dif ** 2
    dst = np.sum(dif, axis=(3, 4, 5))
    dst = np.linalg.norm()

    # Sort and truncate.
    sel = np.argsort(dst, axis=None)
    sel = sel[:n2]

    # Return as row, col
    sel = np.unravel_index(sel, blk.shape[:2])

    return sel
