# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def generate_chessboard(boardsize):
    chessboard = np.zeros(boardsize ** 2, dtype=np.float32)
    chessboard[::2] = 1.
    chessboard = np.reshape(chessboard, (boardsize, boardsize))
    return chessboard


def generate_vhstripes(imgsize, frac_horiz=0.5, frac_grad=None, plot_stripes=False, colour=False):
    frac_grad = 0. if frac_grad is None else frac_grad
    if isinstance(imgsize, int):
        imgsize = (imgsize, imgsize)
    vstripe = np.mod(range(int(imgsize[0] * (1. - (frac_horiz + frac_grad)))), 2) == 0
    vstripes = np.array([vstripe[:, None].astype(np.float32)] * imgsize[1])[..., 0]

    hstripe = np.mod(range(imgsize[1]), 2) == 0
    if colour:
        vstripes = np.stack([1. - vstripes, 0. *vstripes, 0. * vstripes], axis=2)   # add colour channels
        hstripe = np.stack([0. * hstripe, 1.  * hstripe, 0. *hstripe], axis=1)   # add colour
    if imgsize[0] - vstripes.shape[1] > 0:
        n_grad_pix = int(imgsize[0] * frac_grad)
        if not colour:
            vstripes = vstripes[..., None]
            hstripe = hstripe[..., None]
        hstripes = np.array([hstripe[:, None].astype(np.float32)] * (imgsize[0] - vstripes.shape[1] - n_grad_pix))[..., 0, :]
        hstripes = np.transpose(hstripes, (1, 0, 2))
        img_out = np.concatenate([hstripes, vstripes], axis=-2)
        if frac_grad > 0.:
            n_grad_pix_x = n_grad_pix
            n_grad_pix_y = imgsize[1]
            y, x = np.meshgrid(np.arange(n_grad_pix_x), np.arange(n_grad_pix_y))
            xy = np.concatenate((x[..., None], y[..., None]))
            xy = np.transpose(np.reshape(xy, (2, -1)))
            grad = ((xy[:, 0] + xy[:, 1]) % 2).astype(np.float32)
            # grad = (xy[:, 1] % 3/  3 + xy[:, 0] % 3 / 3) / 1.333
            grad = np.reshape(grad, (n_grad_pix_y, n_grad_pix_x))
            if colour:
                grad = np.concatenate([np.zeros_like(grad)[..., None],
                                       np.zeros_like(grad)[..., None],
                                       grad[..., None]], axis=-1)
            else:
                grad = grad[..., None]
            img_out = np.concatenate([img_out, grad], axis=-2)
        if not colour:
            img_out = img_out[..., 0]
    else:
        img_out = vstripes

    if plot_stripes:
        plt.imshow(img_out)
        plt.colorbar()
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.show()
    return img_out
