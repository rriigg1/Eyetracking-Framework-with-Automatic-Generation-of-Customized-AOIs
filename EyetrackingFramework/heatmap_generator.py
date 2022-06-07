import numpy as np
import cv2


def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M


def generate_heatmap(gazepoints, dispsize, background=None, alpha=0.5, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y)

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    background		-	image over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = gwh / 2
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[int(y):int(y + gwh), int(x):int(x + gwh)] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[int(strt):int(dispsize[1] + strt), int(strt):int(dispsize[0] + strt)]

    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = 0
    # normalize
    heatmap -= lowbound
    heatmap[heatmap < 0] = 0
    heatmap *= 255/np.max(heatmap)
    heatmap = heatmap.astype(np.uint8)
    # colormap
    heatmap_c = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_c = heatmap_c.astype(np.float32)

    # assign alpha values
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2BGRA)
    heatmap_c[heatmap<=0, :] = 0
    heatmap_c[heatmap>0, 3] = 255

    # overlay over background image
    if (background is not None):
        img = background
        if (img.shape != heatmap.shape):
            background = np.zeros((dispsize[1], dispsize[0], 4), dtype=np.float32)
            offset = (background.shape[0] - img.shape[0], background.shape[1] - img.shape[1])
            offset = (int(offset[0]/2), int(offset[1]/2))
            background[offset[0]:img.shape[0]+offset[0], offset[1]:img.shape[1]+offset[1], :3] = img
            background[:, :, 3] = 255
            img = background
        img[heatmap>0, :] *= (1-alpha)
        heatmap_c[heatmap>0, :] *= alpha
        heatmap = img + heatmap_c
    else:
        heatmap = heatmap_c

    return heatmap
