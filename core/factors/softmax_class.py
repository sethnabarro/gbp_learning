# coding=utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor


class SoftmaxClassObservationFactor(NonLinearFactor):
    def __init__(self,
                 label,
                 sigma,
                 logit_var_edges,
                 relin_freq=1,
                 init_lin_point=None,
                 kmult=None,
                 N_rob=None,
                 rob_type=None,
                 classes_sub=None):
        super(SoftmaxClassObservationFactor, self).__init__(
            sigma=sigma,
            init_lin_point=init_lin_point,
            relin_freq=relin_freq,
            N_rob=N_rob,
            rob_type=rob_type,
            kmult=kmult)
        self.logit_var_edges = logit_var_edges
        self.one_hot = label
        self.classes_sub = classes_sub

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        """
        Factor energy:
            $E(\mathbf{f}, y) = (\mathbf{1}_y - \mathrm{softmax}(\mathbf{f}))^2 / \sigma^2$
            where
                $\mathbf{f}$ is the vector of logits
                $y$ is integer of the correct class
                $\mathbf{1}_y$ is one-hot vector, where only the $y$th element is on

        Measurement function:
            $h(\mathbf{f}) = \mathrm{softmax}(\mathbf{f})$

        Jacobian:
            $J := \partial h(\mathbf{f}) / \partial \mathbf{f}
                = \mathrm{diag}( \mathrm{softmax}(f) ) - \mathrm{softmax}(f) \mathrm{softmax}(f)^T $
        """
        logits, = self.var0

        if self.classes_sub:
            logits = tf.gather(logits, self.classes_sub, axis=-1)
            obs = tf.gather(self.one_hot, self.classes_sub, axis=-1)
        else:
            obs = self.one_hot

        # Compute Jacobian
        sm = tf.nn.softmax(logits)
        J = tf.linalg.diag(sm) - sm[..., None] * sm[..., None, :]

        # Compute eta and Lambda for linearised factor
        JTlogits = tf.reduce_sum(J * logits[..., None, :], axis=-1)
        h = sm
        eta = tf.reduce_sum(J * (JTlogits - h + obs)[..., None, :], axis=-1) / self.sigma ** 2.
        Lambda = tf.matmul(J, J, transpose_b=True) / self.sigma ** 2.

        # Get incoming messages
        var_to_fac_eta = self.logit_var_edges.var_to_fac_eta
        var_to_fac_Lambda = self.logit_var_edges.var_to_fac_Lambda

        if self.classes_sub:
            var_to_fac_eta = tf.gather(var_to_fac_eta, self.classes_sub, axis=-1)
            var_to_fac_Lambda = tf.gather(var_to_fac_Lambda, self.classes_sub, axis=-1)

        # Get outgoing
        fac_to_var_eta, fac_to_var_Lambda =\
            self.marginalise(factor_plus_mess_eta=eta + var_to_fac_eta,
                             factor_plus_mess_Lambda=Lambda + tf.linalg.diag(var_to_fac_Lambda),
                             factor_eta=eta,
                             factor_Lambda=Lambda)

        # Update edges
        if self.classes_sub:
            # TODO: below without for loop
            fac_to_var_eta_sparse = tf.zeros_like(self.logit_var_edges.fac_to_var_eta)
            fac_to_var_Lambda_sparse = tf.zeros_like(self.logit_var_edges.fac_to_var_Lambda)
            for i, c in enumerate(self.classes_sub):
                fac_to_var_eta_sparse += tf.one_hot(c, depth=self.one_hot.shape[-1])[None] * fac_to_var_eta[..., i:i + 1]
                fac_to_var_Lambda_sparse += tf.one_hot(c, depth=self.one_hot.shape[-1])[None] * fac_to_var_Lambda[..., i:i + 1]
            fac_to_var_eta = fac_to_var_eta_sparse
            fac_to_var_Lambda = fac_to_var_Lambda_sparse
        self.logit_var_edges.fac_to_var_eta = fac_to_var_eta
        self.logit_var_edges.fac_to_var_Lambda = fac_to_var_Lambda

    def energy(self, conn_vars, robust=None, aggregate=True):
        logits, = conn_vars
        if self.classes_sub:
            logits = tf.gather(logits, self.classes_sub, axis=-1)
            obs = tf.gather(self.one_hot, self.classes_sub, axis=-1)
        else:
            obs = self.one_hot
        E = ((obs - tf.nn.softmax(logits, axis=-1)) / self.sigma) ** 2.
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
        edges = [self.logit_var_edges.named_state if named else self.logit_var_edges.state]
        return edges

    def set_edge_messages(self, edges):
        self.logit_var_edges.state = edges[0]
        return edges


    @property
    def state(self):
        return [self.var0, self.get_edge_messages(), self.one_hot]

    @state.setter
    def state(self, new_state):
        self.var0 = new_state[0]
        self.set_edge_messages(new_state[1])
        self.one_hot = new_state[2]

