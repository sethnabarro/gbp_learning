# coding=utf-8
import tensorflow as tf

from core.factors.softmax_class import SoftmaxClassObservationFactor


class BinaryClassObservationFactor(SoftmaxClassObservationFactor):
    def __init__(self,
                 label,
                 sigma,
                 logit_var_edges,
                 relin_freq=1,
                 init_lin_point=None,
                 kmult=None,
                 N_rob=None,
                 rob_type=None):
        super(BinaryClassObservationFactor, self).__init__(
            label=label,
            logit_var_edges=logit_var_edges,
            sigma=sigma,
            init_lin_point=init_lin_point,
            relin_freq=relin_freq,
            N_rob=N_rob,
            rob_type=rob_type,
            kmult=kmult,)
        self.input_var_edges = self.logit_var_edges
        self.sign_vector = self.one_hot * 2. - 1.   # 1 for elem of correct class, -1 otherwise

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        """
        Factor energy:
            $E(\mathbf{f}, y) = (\mathbf{S}_y - \mathbf{f})^2 / \sigma^2$
            where
                $\mathbf{f}$ is the vector of inputs
                $y$ is integer of the correct class
                $\mathbf{S}_y$ is vector, $y$th element is 1, others are -1

        Measurement function:
            $h(\mathbf{f}) = \mathbf{f}$

        Jacobian:
            $J := \partial h(\mathbf{f}) / \partial \mathbf{f}
                = \mathbb{1} $
        """
        inputs, = self.var0

        # Compute Jacobian
        J = tf.ones_like(inputs)

        # Compute eta and Lambda for linearised factor
        eta = self.sign_vector / self.sigma ** 2.
        Lambda = J * J / self.sigma ** 2.

        # # Get incoming messages
        # var_to_fac_eta = self.input_var_edges.var_to_fac_eta
        # var_to_fac_Lambda = self.input_var_edges.var_to_fac_Lambda
        #
        # print(J.shape, var_to_fac_eta.shape, var_to_fac_Lambda.shape, eta.shape, Lambda.shape, self.sign_vector.shape)
        #
        # # Marginals
        # fac_to_var_eta, fac_to_var_Lambda =\
        #     self.marginalise(factor_plus_mess_eta=eta + var_to_fac_eta,
        #                      factor_plus_mess_Lambda=Lambda + tf.linalg.diag(var_to_fac_Lambda),
        #                      factor_eta=eta,
        #                      factor_Lambda=Lambda)
        #
        # tf.debugging.check_numerics(fac_to_var_eta)
        # tf.debugging.check_numerics(fac_to_var_Lambda)

        # Update edges
        self.input_var_edges.fac_to_var_eta = eta
        self.input_var_edges.fac_to_var_Lambda = Lambda

    def energy(self, conn_vars, robust=None, aggregate=True):
        inputs, = conn_vars
        E = ((self.sign_vector - inputs) / self.sigma) ** 2.
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        E = E[..., None]

        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def get_edge_messages(self, named=False):
        edges = [self.input_var_edges.named_state if named else self.input_var_edges.state]
        return edges

    def set_edge_messages(self, edges):
        self.input_var_edges.state = edges[0]
        return edges
