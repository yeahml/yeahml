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
    # cmap = plt.get_cmap("viridis")
    cmap = plt.get_cmap("Spectral_r")  # reverse spectral [0:blue, 1/max:red]
    bounds = np.linspace(-0.01, 1, 80)  # TODO: explain
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # NOTE: this is a "hacky" fix to outputing images that look how'd we'd expect
    # this may cause unexpected future side effects
    if np.amax(mp) <= 1.0:
        im = ax.imshow(
            mp, interpolation="nearest", origin="lower", cmap=cmap, norm=norm
        )
    else:
        im = ax.imshow(
            mp.astype("int"),
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
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


def plot_four_seg(
    sess, output_dim_list, x, preds, seg_prob, X_batch, y_batch, idx, NUMCLASSES
):
    # TODO: this would be better if it could be a generalization
    # where the 2nd figure is a targeted class
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

    # plot each figure
    # dividing by the number of classes is used to convert values to a 0-1 scale
    # rather than being on a 0-num classes scale (from the argmax), which is then easier
    # to plot and visualize
    implot(X_batch[idx, :, :], ax1)  # raw input image
    implot(
        y_seg_prob[idx, :, :].copy() / NUMCLASSES, ax2, True
    )  # segmentation probability
    implot(
        y_prediction[idx, :, :].copy() / NUMCLASSES, ax3, True
    )  # segmentation probability threshold
    implot(
        y_batch[idx, :, :].copy() / NUMCLASSES, ax4, True
    )  # ground truth segmentation

    # improve figure appearance
    plt.grid(False)
    plt.tight_layout()

    # convert to buffer for tensorboard
    buf = convert_to_buf(plt)
    plt.close("all")

    return buf


def plot_three_seg(sess, output_dim_list, x, preds, X_batch, y_batch, idx, NUMCLASSES):
    # TODO: "8" should be ['num_classes']

    # TODO: TEMP
    y_softmax = sess.run(preds, {x: X_batch})
    y_prediction = np.argmax(y_softmax[0], axis=3)

    # create figure
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))

    # plot each figure
    # dividing by the number of classes is used to convert values to a 0-1 scale
    # rather than being on a 0-num classes scale (from the argmax), which is then easier
    # to plot and visualize
    implot(X_batch[idx, :, :], ax1)  # raw input image
    implot(
        y_prediction[idx, :, :].copy() / NUMCLASSES, ax2, True
    )  # segmentation probability threshold
    implot(
        y_batch[idx, :, :].copy() / NUMCLASSES, ax3, True
    )  # ground truth segmentation

    # improve figure appearance
    plt.grid(False)
    plt.tight_layout()

    # convert to buffer for tensorboard
    buf = convert_to_buf(plt)
    plt.close("all")

    return buf
