# coding=utf-8
import tensorflow as tf
import time

from core.factors import Factor
from core.utils.utils import tf_fn_wrapper


class TestFactor(tf.test.TestCase):
    def ground_truth_marginals(self, fac_eta, msg_eta, msg_Lambda, fac_J_div_sigma=None, fac_Lambda=None):
        if fac_Lambda is None:
            fac_Lambda = fac_J_div_sigma[..., None] * fac_J_div_sigma[..., None, :]

        fac_Lambda_mess = fac_Lambda + tf.linalg.diag(msg_Lambda)
        fac_eta_mess = fac_eta + msg_eta

        n_vars = fac_eta.shape[-1]
        eta_margs = tf.TensorArray(dtype=tf.float32, size=n_vars)
        Lambda_margs = tf.TensorArray(dtype=tf.float32, size=n_vars)
        for v in range(n_vars):
            not_v = [r for r in list(range(n_vars)) if r != v]
            fac_Lambda_mess_b = tf.gather(fac_Lambda_mess, not_v, axis=-2)
            fac_Lambda_mess_bb = tf.gather(fac_Lambda_mess_b, not_v, axis=-1)
            fac_Lambda_mess_bb_inv = tf.linalg.inv(fac_Lambda_mess_bb)
            fac_Lambda_mess_ba = fac_Lambda_mess_b[..., v][..., None]
            fac_Lambda_mess_ab = tf.linalg.matrix_transpose(fac_Lambda_mess_ba)
            fac_eta_b = tf.gather(fac_eta_mess, not_v, axis=-1)
            eta_marg = fac_eta[..., v] - tf.matmul(tf.matmul(fac_Lambda_mess_ab, fac_Lambda_mess_bb_inv), fac_eta_b[..., None])[..., 0, 0]
            Lambda_marg = fac_Lambda[..., v, v] - tf.matmul(tf.matmul(fac_Lambda_mess_ab, fac_Lambda_mess_bb_inv), fac_Lambda_mess_ba)[..., 0, 0]
            eta_margs = eta_margs.write(v, eta_marg)
            Lambda_margs = Lambda_margs.write(v, Lambda_marg)

        ndim = len(fac_eta.shape.as_list())
        return tf.transpose(eta_margs.stack(), list(range(1, ndim)) + [0]), \
            tf.transpose(Lambda_margs.stack(), list(range(1, ndim)) + [0])

    def sherman_morrison_marginals(self, inmsg_eta, inmsg_Lambda, fac_eta, fac_J_div_sigma,
                                   static_graph=False, batchsize=1, optimised=False, optimised_no_loops=True):
        def _marg():
            if optimised:
                return Factor.marginalise_sherman_morrison_old(mess_eta=inmsg_eta,
                                                               mess_Lambda=inmsg_Lambda,
                                                               factor_eta=fac_eta,
                                                               J_div_sigma=fac_J_div_sigma,
                                                               batchsize=batchsize)
            if optimised_no_loops:
                return Factor.marginalise_sherman_morrison(mess_eta=inmsg_eta,
                                                           mess_Lambda=inmsg_Lambda,
                                                           factor_eta=fac_eta,
                                                           J_div_sigma=fac_J_div_sigma,
                                                           batchsize=batchsize)
            else:
                return Factor.marginalise_sherman_morrison_older(mess_eta=inmsg_eta,
                                                                 mess_Lambda=inmsg_Lambda,
                                                                 factor_eta=fac_eta,
                                                                 J_div_sigma=fac_J_div_sigma)

        maybe_compiled = tf_fn_wrapper(_marg, use_static_graph=static_graph, xla_compile=False)
        if static_graph:
            maybe_compiled()   # Compile it at run time so have to run once
        tbef = time.time()
        # tf.profiler.experimental.start('profile/')
        # for _ in range (10):
        out = maybe_compiled()
        # tf.profiler.experimental.stop()
        taft = time.time()

        return out, taft - tbef

    def test_sherman_morrison_marg_speed(self):
        # Generate some invertible matrices (diagonally dominant)
        batch_dims = 20, 50, 50, 16
        nvars = 32
        sig = 2.
        tf.random.set_seed(89)
        J_div_sig = tf.random.normal(batch_dims + (nvars,)) / sig
        eta = tf.random.normal(batch_dims + (nvars,))
        inmsg_Lambda = tf.maximum(tf.abs(tf.random.normal(batch_dims + (nvars,))) * 2., 1.)
        inmsg_eta = tf.random.normal(batch_dims + (nvars,))

        # Do Sherman morrison inversion. Both with old and optimised code
        (out_eta, out_Lambda), time_taken = \
            self.sherman_morrison_marginals(inmsg_eta, inmsg_Lambda, eta, J_div_sig, static_graph=True,
                                            optimised=False, optimised_no_loops=False)
        (out_eta_opt, out_Lambda_opt), time_taken_opt = \
            self.sherman_morrison_marginals(inmsg_eta, inmsg_Lambda, eta, J_div_sig,
                                            static_graph=True, batchsize=2, optimised=True, optimised_no_loops=False)

        (out_eta_opt_no_loop, out_Lambda_opt_no_loop), time_taken_opt_no_loop = \
            self.sherman_morrison_marginals(inmsg_eta, inmsg_Lambda, eta, J_div_sig,
                                            static_graph=True, optimised=False, optimised_no_loops=True)

        # check old method and new method give same results
        self.assertAllClose(out_eta_opt, out_eta, atol=0.01)
        self.assertAllClose(out_Lambda_opt, out_Lambda, atol=0.01)

        self.assertAllClose(out_Lambda_opt_no_loop, out_Lambda, atol=0.01)
        self.assertAllClose(out_eta_opt_no_loop, out_eta, atol=0.01)

        # Print timings
        print(f'Old Sherman-Morrison marginalisation took {time_taken}, '
              f'optimised took {time_taken_opt}, ' 
              f'no loop optimised took {time_taken_opt_no_loop}')

    def test_sherman_morrison_marg_accuracy(self):
        batch_dims = 2, 50, 50, 16
        nvars = 8
        sig = 3.
        tf.random.set_seed(89)
        J_div_sig = tf.random.normal(batch_dims + (nvars,)) / sig
        eta = tf.random.normal(batch_dims + (nvars,))
        inmsg_Lambda = tf.maximum(tf.abs(tf.random.normal(batch_dims + (nvars,))) * 3., 1.)
        inmsg_eta = tf.random.normal(batch_dims + (nvars,))

        # Do straightforward (but slow) marginalisation
        out_eta_gt, out_Lambda_gt = self.ground_truth_marginals(eta, inmsg_eta, inmsg_Lambda, fac_J_div_sigma=J_div_sig)

        # Check direct marg gives same results as sherman morrison
        (out_eta, out_Lambda), time_taken = \
            self.sherman_morrison_marginals(inmsg_eta, inmsg_Lambda, eta, J_div_sig,
                                            static_graph=True, optimised=False, optimised_no_loops=False)
        (out_eta_opt, out_Lambda_opt), time_taken_opt = \
            self.sherman_morrison_marginals(inmsg_eta, inmsg_Lambda, eta, J_div_sig, static_graph=True,
                                            optimised=True, optimised_no_loops=False)
        (out_eta_opt_no_loop, out_Lambda_opt_no_loop), time_taken_opt_no_loop = \
            self.sherman_morrison_marginals(inmsg_eta, inmsg_Lambda, eta, J_div_sig,
                                            static_graph=True, optimised=False, optimised_no_loops=True)

        self.assertAllClose(out_eta_opt, out_eta_gt, atol=0.01)
        self.assertAllClose(out_Lambda_opt, out_Lambda_gt, atol=0.01)
        self.assertAllClose(out_Lambda_opt, out_Lambda, atol=0.01)
        self.assertAllClose(out_eta_opt, out_eta, atol=0.01)
        self.assertAllClose(out_Lambda_opt_no_loop, out_Lambda, atol=0.01)
        self.assertAllClose(out_eta_opt_no_loop, out_eta, atol=0.01)

    def runTest(self):
        self.test_sherman_morrison_marg_speed()
        self.test_sherman_morrison_marg_accuracy()


if __name__ == '__main__':
    tf.test.TestCase.run(TestFactor())
    # TestFactor().test_sherman_morrison_marg_speed()
