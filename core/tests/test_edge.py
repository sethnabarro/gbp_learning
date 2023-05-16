# coding=utf-8
import numpy as np
import tensorflow as tf

from core.edge import Edge

# Made up edge messages to check msg diffs calculated correctly
EDGE_MSGS = {'fac_to_var_eta': tf.convert_to_tensor([[-10., 0.],
                                                     [-1., -1.],
                                                     [2., -1.],
                                                     [4., 4.],
                                                     [0., -1.],
                                                     [2., -1.]]),
             'fac_to_var_Lambda': tf.convert_to_tensor([[0., 1.],
                                                        [1.5, 1.],
                                                        [2., 1.],
                                                        [1.5, 0.6],
                                                        [1.5, 0.6],
                                                        [1.5, 0.7]]),
             'var_to_fac_eta': tf.convert_to_tensor([[-5., 0.],
                                                     [-0.2, -0.3],
                                                     [-0.3, -0.3],
                                                     [-2., 0.1],
                                                     [-1.5, 0.4],
                                                     [-1.5, 0.405]]),
             'var_to_fac_Lambda': tf.convert_to_tensor([[0., 2.],
                                                        [0.3, 0.8],
                                                        [0.25, 0.98],
                                                        [0.5, 0.9],
                                                        [0.5, 0.9],
                                                        [0.5, 0.9]])}
MESSAGE_TYPES = ('var_to_fac_eta', 'var_to_fac_Lambda', 'fac_to_var_eta', 'fac_to_var_Lambda')


def run_edge_updates_eager(edg, msg_types=None):
    n_iters = EDGE_MSGS['var_to_fac_eta'].shape[0]
    msg_types = msg_types or EDGE_MSGS.keys()
    for itr in range(n_iters):
        for mt in msg_types:
            setattr(edg, mt, EDGE_MSGS[mt][itr])
    return edg


def edge_update_iter(msg_types, edg: Edge, itr, v2f_eta, v2f_Lambda, f2v_eta, f2v_Lambda,
                     v2f_eta_diff=None, v2f_Lambda_diff=None, f2v_eta_diff=None, f2v_Lambda_diff=None):
    edge_state = [itr, v2f_eta, v2f_Lambda, f2v_eta, f2v_Lambda]
    if edg.track_msg_diffs:
        edge_state += [v2f_eta_diff, v2f_Lambda_diff, f2v_eta_diff, f2v_Lambda_diff]
    edg.state = edge_state
    for mt in msg_types:
        setattr(edg, mt, EDGE_MSGS[mt][edg.iter[mt][0]])
    return edg.state


def run_edge_updates_static(edg: Edge, msg_types=None):
    n_iters = EDGE_MSGS['var_to_fac_eta'].shape[0]
    msg_types = msg_types or MESSAGE_TYPES

    # We already did one iteration when initialising edges
    edg.iter = {k: tf.zeros([1], dtype=tf.int32) for k in edg.iter}

    def _edge_update_iter(*args):
        return edge_update_iter(msg_types, edg, *args)

    @tf.function
    def _run_updates():
        edg.n_iters = n_iters
        edg.init_msg_diff_recorder()
        loop_vars = list(edg.state)

        msgs_final = \
            tf.while_loop(cond=lambda itr, *_: tf.less(tf.reduce_max(itr), n_iters),
                          body=_edge_update_iter,
                          loop_vars=loop_vars)

        edg.state = msgs_final

        return msgs_final

    edge_state = _run_updates()

    edg.state = edge_state


class TestEdge(tf.test.TestCase):
    def setUp(self, **kwargs):
        # Append init edge msgs to kwargs
        for k, v in EDGE_MSGS.items():
            kwargs[k] = tf.zeros_like(v[[0]])

        self.edge = Edge(**kwargs)

    def test_iter_incrementer_eager(self):
        # Update all edge msgs same number of times
        run_edge_updates_eager(self.edge)
        n_iters_true = EDGE_MSGS['var_to_fac_eta'].shape[0]
        for itr_counts in self.edge.iter.values():
            self.assertEqual(itr_counts, n_iters_true)

        # Try with update only var_to_fac_eta
        self.setUp()  # Reinit edge
        msgs_to_update = ('var_to_fac_eta',)
        run_edge_updates_eager(self.edge, msg_types=msgs_to_update)
        n_iters_true = EDGE_MSGS['var_to_fac_eta'].shape[0]
        for nm, itr_counts in self.edge.iter.items():
            self.assertEqual(itr_counts, n_iters_true if nm in msgs_to_update else 0)

    def test_iter_incrementer_static(self):
        # Update all edges same number of times
        run_edge_updates_static(self.edge)
        n_iters_true = EDGE_MSGS['var_to_fac_eta'].shape[0]
        for itr_counts in self.edge.iter.values():
            self.assertEqual(itr_counts, n_iters_true)

        # Try with update only var_to_fac_eta
        self.setUp()  # Reinit edge
        msgs_to_update = ('var_to_fac_eta',)
        run_edge_updates_static(self.edge, msg_types=msgs_to_update)
        n_iters_true = EDGE_MSGS['var_to_fac_eta'].shape[0]
        for nm, itr_counts in self.edge.iter.items():
            self.assertEqual(itr_counts, n_iters_true if nm in msgs_to_update else 0)

    def test_msg_diffs_eager(self):
        # Check tracking only the most recent diff is accurate
        n_iters = EDGE_MSGS['fac_to_var_eta'].shape[0]
        self.setUp(track_msg_diffs=True, record_all_diffs=False, n_iters=n_iters)
        run_edge_updates_eager(self.edge)
        for k, v in self.edge.msg_diffs.items():
            last_diff = tf.reduce_max(tf.abs(EDGE_MSGS[k][[-1]] - EDGE_MSGS[k][[-2]]))
            self.assertEqual(last_diff, v)

        self.setUp(track_msg_diffs=True, record_all_diffs=True, n_iters=n_iters)
        run_edge_updates_eager(self.edge)
        for k, v in self.edge.msg_diffs.items():
            diffs = [tf.reduce_max(tf.abs(EDGE_MSGS[k][0]))]
            diffs += [tf.reduce_max(tf.abs(EDGE_MSGS[k][[i]] - EDGE_MSGS[k][[i - 1]])) for i in range(1, n_iters)]
            self.assertAllEqual(tf.convert_to_tensor(diffs), v)

    def test_msg_diffs_static(self):
        n_iters = EDGE_MSGS['fac_to_var_eta'].shape[0]
        self.setUp(track_msg_diffs=True, record_all_diffs=False, n_iters=n_iters)
        run_edge_updates_static(self.edge)
        for k, v in self.edge.msg_diffs.items():
            last_diff = tf.reduce_max(tf.abs(EDGE_MSGS[k][[-1]] - EDGE_MSGS[k][[-2]]))
            self.assertEqual(last_diff, v)

        self.setUp(track_msg_diffs=True, record_all_diffs=True, n_iters=n_iters)
        run_edge_updates_static(self.edge)
        for k, v in self.edge.msg_diffs.items():
            diffs = [tf.reduce_max(tf.abs(EDGE_MSGS[k][0]))]
            diffs += [tf.reduce_max(tf.abs(EDGE_MSGS[k][[i]] - EDGE_MSGS[k][[i - 1]])) for i in range(1, n_iters)]
            self.assertAllEqual(tf.convert_to_tensor(diffs), v)

    def test_apply_momtm_eager(self):
        n_iters = EDGE_MSGS['fac_to_var_eta'].shape[0]
        mom = 0.9
        self.setUp(track_msg_diffs=True, record_all_diffs=False, n_iters=n_iters, momentum=mom)

        # Momentum only applied to fac -> var messages
        msgs_with_mom = ('fac_to_var_eta', 'fac_to_var_Lambda')
        run_edge_updates_eager(self.edge)
        for mtype in MESSAGE_TYPES:
            if mtype in msgs_with_mom:
                # Calculate correct final message with mom
                # First iter - set message directly (no mom)
                mess_corr = EDGE_MSGS[mtype][0]

                # Subsequent iters - apply mom
                for i in range(n_iters):
                    mess_corr = mom * mess_corr + (1. - mom) * EDGE_MSGS[mtype][i]
            else:
                mess_corr = EDGE_MSGS[mtype][-1]
            self.assertAllEqual(mess_corr, getattr(self.edge, mtype))

    def test_apply_momtm_static(self):
        n_iters = EDGE_MSGS['fac_to_var_eta'].shape[0]
        mom = 0.9
        self.setUp(track_msg_diffs=True, record_all_diffs=False, n_iters=n_iters, momentum=mom)

        # Momentum only applied to fac -> var messages
        msgs_with_mom = ('fac_to_var_eta', 'fac_to_var_Lambda')
        run_edge_updates_static(self.edge)
        for mtype in MESSAGE_TYPES:
            if mtype in msgs_with_mom:
                # Calculate correct final message with mom
                # First iter - set message directly (no mom)
                mess_corr = EDGE_MSGS[mtype][0]

                # Subsequent iters - apply mom
                for i in range(n_iters):
                    mess_corr = mom * mess_corr + (1. - mom) * EDGE_MSGS[mtype][i]
            else:
                mess_corr = EDGE_MSGS[mtype][-1]
            self.assertAllEqual(mess_corr, getattr(self.edge, mtype))


if __name__ == '__main__':
    tf.test.main()


