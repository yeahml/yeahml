# creating plots for tensorboard
import numpy as np
import tensorflow as tf
import io
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable  # colorbar helper


def convert_to_buf(plt):
    """convert a plt to a buffer to be displayed in TensorBoard
    
    Arguments:
        plt {matplotlib.pyplot} -- the completed plot to be displayed in TensorBoard
    
    Returns:
        [io.BytesIO] -- a bytes buffer of a plot
    """
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf


def implot(mp, ax, SHOW_CB=False):
    cmap = plt.get_cmap("viridis")
    bounds = np.linspace(-0.01, 1, 80)  # TODO: explain
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    im = ax.imshow(mp, interpolation="nearest", origin="lower", cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if SHOW_CB:
        if np.min(mp) != np.max(mp):
            cbar = plt.colorbar(im, cax=cax, format="%1.2f", boundaries=bounds)
        else:
            cax.set_axis_off()
    else:
        cax.set_axis_off()

    ax.set_axis_off()


def plot_four_segmentation_array(
    sess, output_dim_list, x, preds, seg_prob, X_batch, y_batch, idx
):
    # TODO: TEMP
    y_softmax, y_seg_prob = sess.run([preds, seg_prob], {x: X_batch})
    y_prediction = np.argmax(y_softmax[0], axis=3)

    # convert widthxheightx1 to widthxheight
    y_seg_prob = y_seg_prob[0].reshape(
        len(X_batch), output_dim_list[1], output_dim_list[2]
    )

    # create figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        nrows=1, ncols=4, sharey=True, figsize=(12, 4)
    )
    implot(X_batch[idx, :, :], ax1)  # raw input image
    implot(y_seg_prob[idx, :, :], ax2, True)  # segmentation probability
    implot(y_prediction[idx, :, :], ax3, True)  # segmentation probability threshold
    implot(y_batch[idx, :, :], ax4, True)  # ground truth segmentation

    # improve figure appearance
    plt.grid(False)
    plt.tight_layout()

    # convert to buffer for tensorboard
    buf = convert_to_buf(plt)

    return buf
