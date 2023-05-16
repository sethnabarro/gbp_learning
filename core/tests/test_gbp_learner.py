import numpy as np
import matplotlib.pyplot as plt
import unittest

from core.gbp_learner_ import GBPLearner
from core.utils.utils import patchify_image


class TestImgModel(unittest.TestCase):
    def test_calculate_pixel_marginals(self):
        # TODO: FIX BELOW FOR CLASS METHOD!
        recfield = 3
        img = np.repeat(np.array([float(x % 2 == 0) for x in range(20)])[None], axis=0, repeats=20)
        gbp = GBPLearner(img,None, None,None,None,None,None)
        recon = gbp.get_pixel_marginal(patchify_image(img, recfield), img)

        fig, axs = plt.subplots(1, 2)

        im = axs[0].imshow(img)
        fig.colorbar(im, ax=axs[0])
        img = img[None, ..., None]
        # recon = calculate_pixel_marginals(np.zeros_like(img), np.zeros_like(img),
        #                               patchify_image(img, recfield), patchify_image(img, 3))[0]
        im = axs[1].imshow(recon[0,..., 0] / (recfield ** 2))
        fig.colorbar(im, ax=axs[1])
        plt.tight_layout()
        plt.show()

        # Should be equal except around border
        np.testing.assert_allclose(img[0, 2:-2, 2:-2, 0],
                                   recon[0, 2:-2, 2:-2, 0] / (recfield ** 2))
