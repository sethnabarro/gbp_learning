# coding=utf-8
import shutil
import os
import tensorflow as tf

from core.inference.gbp.learner import GBPLearner
from core.utils.saveload import save_gbp_net, load_gbp_net, assert_model_marginals_equal
from core.utils.utils import dotdict as dd
from experiments.utils.exp_setup_utils import init_layers
from inputs.synthetic import generate_vhstripes


def get_args():
    recon_factor_args = dd(additive_factor=True,
                           sigma=1.,
                           N_rob=4.,
                           rob_type='tukey',
                           sum_filters=True,
                           kmult=1.,
                           decompose=True,
                           relin_freq=1,
                           use_bias=True,
                           relative_to_centre=False,
                           ksize=3,
                           stride=1)
    pixel_obs_factor = dd(sigma=0.1,
                          N_rob=10.,
                          rob_type='tukey',
                          kmult=1.,
                          relin_freq=1)
    avg_pool_factor = dd(sigma=0.1, ksize=2)
    filter_prior_factor = dd(sigma=1., mean=0.)
    coeff_prior_factor = dd(sigma=1.,
                            mean=0.)
    bias_prior_factor = dd(mean=0., sigma=1.)
    dense_factor = dd(sigma=0.2, relin_freq=1)
    factors = dd(recon=recon_factor_args,
                 dense=dense_factor,
                 weight_prior=filter_prior_factor,
                 bias_prior=bias_prior_factor,
                 pixel_obs=pixel_obs_factor,
                 coeff_prior=coeff_prior_factor,
                 avg_pool=avg_pool_factor)

    netconf = [dd(name='conv1', type='conv', n_filters=2),
               dd(name='avg_pool2', type='avg_pool'),
               dd(name='conv3', type='conv', n_filters=2),
               dd(name='dense1', type='dense', outdim=3)]
    config = dd(inference='gbp',
                architecture=netconf,
                logdir='tmp/checkpoints/',
                factors=factors,
                n_iters_per_train_batch=20,
                deterministic_init=False,
                momentum=0.6,
                dropout=0.,
                use_filter_coeffs=True,
                use_component_vars=False,
                random_coeff_init=True,
                init_weight_std=1.,
                init_coeff_std=1.,
                weight_init_seed=666,
                coeff_init_seed=999)
    return config


def get_model(img, args):
    lays = init_layers(args, img,
                       weight_init_std=args.init_weight_std,
                       weight_init_seed=888,
                       coeff_init_seed=666)
    model = GBPLearner(layers=lays)
    return model


class TestSaveLoad(tf.test.TestCase):
    def init_model(self):
        img1 = generate_vhstripes((12, 12))[..., None]
        img2 = generate_vhstripes((12, 12), frac_horiz=0.8)[..., None]
        self.img = tf.concat([img1[None], img2[None]], axis=0)
        self.args = get_args()
        self.model = get_model(self.img, self.args)

    def save_model(self, params_only=False):
        # Write the current model to file
        save_gbp_net(gbp_net=self.model,
                     savedir=self.args.logdir,
                     args=self.args,
                     n_batch_so_far=0,
                     n_iter_current_batch=5,
                     input_image=self.img,
                     params_only=params_only)

    def test_saving(self):
        """Check saving code runs without error"""
        # Create temp dir to save model to
        os.mkdir('tmp/')
        self.init_model()

        try:
            # Run a few GBP iterations
            self.model.run_inference(5)
            self.save_model()
        finally:
            shutil.rmtree('tmp/')

    def test_saving_params_only(self):
        """Check saving code runs without error"""
        # Create temp dir to save model to
        os.mkdir('tmp/')
        self.init_model()

        try:
            # Run a few GBP iterations
            self.model.run_inference(5)
            self.save_model(params_only=True)
        finally:
            shutil.rmtree('tmp/')

    def test_loading(self):
        """Check loading code runs without error"""
        # First create model and save it
        os.mkdir('tmp/')
        self.init_model()
        try:
            # Run a few GBP iterations
            self.model.run_inference(5)
            self.save_model()
            loaddir = get_args().logdir
            model_loaded = load_gbp_net(loaddir)

            # Check marginals of the loaded model same as original
            assert_model_marginals_equal(model_loaded, self.model)

            # Check edges are correct -
            # run a few more iters of GBP in both models, check marginals still match
            # They will only match if edge states were saved and loaded correctly
            self.model.run_inference(8)
            model_loaded.run_inference(8)

            # Check marginals of both models agree
            assert_model_marginals_equal(model_loaded, self.model)

            # Check that marginals do not agree if run one for more iters
            self.model.run_inference(7)
            model_loaded.run_inference(8)

            # Check marginals do not agree
            def _should_raise_exception():
                assert_model_marginals_equal(model_loaded, self.model)
            self.assertRaises(tf.errors.InvalidArgumentError, _should_raise_exception)

        finally:
            shutil.rmtree('tmp/')

    def test_loading_params_only(self):
        """Check loading code runs without error"""
        # First create model and save it
        os.mkdir('tmp/')
        self.init_model()
        try:
            # Run a few GBP iterations
            self.model.run_inference(5)
            self.save_model(params_only=True)
            loaddir = get_args().logdir
            model_loaded = load_gbp_net(loaddir)

            # Check marginals of the params in loaded model same as original
            assert_model_marginals_equal(model_loaded, self.model, params_only=True)

            # Check that marginals do not agree if do some more GBP
            # (We didn't save and load edges therefore the model params should diverge)
            self.model.run_inference(5)
            model_loaded.run_inference(5)

            # Check marginals do not agree
            def _should_raise_exception():
                assert_model_marginals_equal(model_loaded, self.model, params_only=True)
            self.assertRaises(tf.errors.InvalidArgumentError, _should_raise_exception)

        finally:
            shutil.rmtree('tmp/')


if __name__ == '__main__':
    TestSaveLoad().run()
