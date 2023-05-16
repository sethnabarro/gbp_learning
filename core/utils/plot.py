# coding=utf-8
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from core.inference.gbp.layers import GBPPoolLayer, GBPConvLayer, GBPDenseLayer, GBPSoftmaxClassObservationLayer
from core.utils.utils import match_edgenames_without_bs


imshow = lambda *args, **kwargs: plt.imshow(*args, **kwargs, interpolation='none')


def plot_model_coeffs(model, plot_dir, itr=None, max_imgs=None, y_class=None):
    batchsize = model.layers[0].coeff_vars.mu.shape[0]
    max_imgs = max_imgs or batchsize
    in_determ = None
    for layid, lay in enumerate(model.layers):
        if isinstance(lay, (GBPConvLayer, GBPPoolLayer)):
            for i in range(min(max_imgs, batchsize)):
                plot_coeffs_subplots(lay.coeff_vars.mu[i],
                                     itr=itr,
                                     plotfile=os.path.join(plot_dir, f'layer_{layid}', f'img_mu_{i}.png'))
                plot_coeffs_subplots(lay.coeff_vars.sigma[i],
                                     itr=itr,
                                     plotfile=os.path.join(plot_dir, f'layer_{layid}', f'img_std_{i}.png'))
                if hasattr(lay, 'use_feedforward_factor'):
                    if lay.use_feedforward_factor:
                        # lay.relinearise_factors(0)
                        coeffs_determ = lay.filter_factor.forward_deterministic(in_determ)
                        plot_coeffs_subplots(coeffs_determ[i],
                                             itr=itr,
                                             plotfile=os.path.join(plot_dir, f'layer_{layid}', f'img_{i}_determ.png'))
                        in_determ = coeffs_determ

        elif isinstance(lay, GBPDenseLayer):
            plotfn = os.path.join(plot_dir, f'layer_{layid}', f'dense_inout_mu.png')
            if itr is not None:
                plotfn = plotfn.replace('_inout_', f'_inout_itr{itr}_')
            # Flatten input vars
            inp_mu_flat = tf.reshape(lay.input_vars.mu, [lay.input_vars.mu.shape[0], -1])
            inp_std_flat = tf.reshape(lay.input_vars.sigma, [lay.input_vars.sigma.shape[0], -1])
            if y_class is None:
                if len(model.layers) == layid:
                    if isinstance(model.layers[layid + 1], GBPSoftmaxClassObservationLayer):
                        y_class = model.layers[layid + 1].softmax_factor.one_hot
            plot_dense_input_output(inp_mu_flat, lay.coeff_vars.mu, plotfn, y_class)
            plot_dense_input_output(inp_std_flat,
                                    lay.coeff_vars.sigma,
                                    plotfn.replace('_mu.', '_std.'),
                                    y_class)
            if in_determ is not None:
                coeffs_determ = lay.dense_factor.forward_deterministic(in_determ)
                plot_dense_input_output(inp_mu_flat,
                                        coeffs_determ,
                                        plotfn.replace('_mu.', '_determ.'),
                                        y_class)
                in_determ = coeffs_determ


def plot_model_weights(model, plot_dir, itr=None):
    for layid, lay in enumerate(model.layers):
        if isinstance(lay, GBPConvLayer):
            fname = f'mu.png' if itr is None else f'itr{itr}_mu.png'
            plot_filters(lay.filter_vars.mu,
                         itr=itr,
                         plotfile=os.path.join(plot_dir, f'layer_{layid}', fname))
            plot_filters(lay.filter_vars.sigma,
                         itr=itr,
                         plotfile=os.path.join(plot_dir, f'layer_{layid}', fname.replace('_mu.', '_std.')))
        elif isinstance(lay, GBPDenseLayer):
            fname = f'weights_mu.png' if itr is None else f'weights_itr{itr}_mu.png'
            plot_dense_weights(lay.weight_vars.mu,
                               plotfile=os.path.join(plot_dir, f'layer_{layid}', fname))
            plot_dense_weights(lay.weight_vars.sigma,
                               plotfile=os.path.join(plot_dir, f'layer_{layid}', fname.replace('_mu.', '_std.')))


def plot_message_convergence(model, plot_dir, edge_stats=None, markers=None, logscale=True):
    n_col, n_row = 2, 2
    min_diff = 1e-7 if logscale else np.finfo(np.float32).min
    for l_id, lay in enumerate(model.layers):
        for e, ed in enumerate(lay.edges):
            fig, axs = plt.subplots(n_col, n_row)
            for m_id, mtype in enumerate(ed.msg_diffs):
                if edge_stats is None:
                    to_plot = ed.msg_diffs[mtype]
                else:
                    if str(ed) in edge_stats[l_id]:
                        to_plot = edge_stats[l_id][str(ed)][mtype]
                    else:
                        # Difference in batch size
                        en = match_edgenames_without_bs(str(ed), list(edge_stats[l_id].keys()))
                        to_plot = edge_stats[l_id][en][mtype]

                axs[m_id % 2, m_id // n_row].plot(tf.maximum(to_plot, min_diff), linewidth=0.5)
                axs[m_id % 2, m_id // n_row].set_ylabel(mtype)
                axs[m_id % 2, m_id // n_row].set_xlabel('Iters')
                if logscale:
                    axs[m_id % 2, m_id // n_row].set_yscale('log')
                if markers is not None:
                    axs[m_id % 2, m_id // n_row].scatter(markers, [int(logscale) for _ in markers], marker='^')
            plt.tight_layout()
            fig.set_size_inches(14, 14)
            plt.savefig(os.path.join(plot_dir, f'layer_{l_id}', f'edge_{ed}.png'))
            plt.close('all')

    fig, axs = plt.subplots(n_col, n_row)
    cmap = cm.get_cmap('viridis', len(model.layers))
    for l_id, lay in enumerate(model.layers):
        for e, ed in enumerate(lay.edges):
            for m_id, mtype in enumerate(ed.msg_diffs):
                if str(ed) in edge_stats[l_id]:
                    to_plot = edge_stats[l_id][str(ed)][mtype]
                else:
                    # Difference in batch size
                    en = match_edgenames_without_bs(str(ed), list(edge_stats[l_id].keys()))
                    to_plot = edge_stats[l_id][en][mtype]
                axs[m_id % 2, m_id // n_row].plot(tf.maximum(to_plot, min_diff),
                                                  color=cmap(l_id), linewidth=0.5)
                axs[m_id % 2, m_id // n_row].set_ylabel(mtype)
                axs[m_id % 2, m_id // n_row].set_xlabel('Iters')
                if logscale:
                    axs[m_id % 2, m_id // n_row].set_yscale('log')
                if markers is not None:
                    axs[m_id % 2, m_id // n_row].scatter(markers, [int(logscale) for _ in markers], marker='^')
    plt.tight_layout()
    fig.set_size_inches(14, 14)
    plt.savefig(os.path.join(plot_dir, f'all_edges.png'))
    plt.close('all')


def plot_model_diagnostics(model,
                           itr_str,
                           plot_dir,
                           edge_stats=None,
                           plot_coeffs=True,
                           plot_weights=True,
                           plot_msg_convergence=True,
                           y_class=None):
    if plot_coeffs:
        print('Plotting coefficients.')
        plot_model_coeffs(model, plot_dir=os.path.join(plot_dir, 'coeffs'), itr=itr_str, max_imgs=3, y_class=y_class)
    if plot_weights:
        print('Plotting weights.')
        plot_model_weights(model, plot_dir=os.path.join(plot_dir, 'weights'), itr=itr_str)
    if plot_msg_convergence:
        has_msg = edge_stats is not None or model.layers[0].edges[0].msg_diffs is not None
        if has_msg:
            print('Plotting message convergence.')
            plot_message_convergence(model,
                                     edge_stats=edge_stats,
                                     plot_dir=os.path.join(plot_dir, 'msg_diffs'))


def plot_denoising(img_corrupt, img_denoised, itr,
                   title=None,
                   img=None,  # Clean
                   figsize=None,
                   plotfile=None,
                   separate_plots=False,
                   dpi=800):
    """Plots original noisy image against denoised image"""
    plt.close('all')
    is_colour = img_corrupt.shape[-1] == 3

    def _plot_im(img_pl, ax=None, f=None):
        cbar_args = dict(fraction=0.04, pad=0.04) if separate_plots else dict(fraction=0.02, pad=0.04)
        if ax is None:
            plt.close('all')
            ax = plt.gca()
        if f is None:
            f = plt.gcf()
            if separate_plots:
                f.set_size_inches(4, 4)

        im = ax.imshow(img_pl if is_colour else img_pl[..., 0], interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        im.set_clim(np.min(img_pl), np.max(img_pl))
        if not is_colour:
            cbar = f.colorbar(im, ax=ax, **cbar_args)
            if separate_plots:
                cbar.ax.tick_params(labelsize=15)
        return im

    title = title or ""
    plot_se = img is not None
    if separate_plots:
        _plot_im(img_corrupt[0])
        plt.savefig(plotfile.replace('.p', '_corrupt.p'), bbox_inches='tight', dpi=dpi)
        _plot_im(img_denoised[0])
        plt.savefig(plotfile.replace('.p', '_denoised.p'), bbox_inches='tight', dpi=dpi)
        if plot_se:
            _plot_im(np.sqrt((img_denoised[0] - img[0]) ** 2.))
            plt.savefig(plotfile.replace('.p', '_sqerr.p'), bbox_inches='tight', dpi=dpi)
    else:
        fs = (10 + 3. * int(plot_se), 3.5) if figsize is None else figsize
        fig, axs = plt.subplots(1, 2 + int(plot_se), figsize=fs)
        _plot_im(img_corrupt[0], ax=axs[0], f=fig)
        _plot_im(img_denoised[0], ax=axs[1], f=fig)
        axs[1].set_title(f'{title} Iter: {itr}')
        if plot_se:
            axs[2].set_title(f'Abs Error')
            _plot_im(np.abs(img_denoised[0] - img[0]), ax=axs[2], f=fig)
        plt.tight_layout()
        if plotfile is None:
            plt.show()
        else:
            plt.savefig(plotfile, bbox_inches='tight')
        plt.close('all')


def plot_component_energies(components, weights, filters, comp_energy, itr=None, N_rob_comp=None, plotfile=None):
    E_comp = comp_energy([filters, weights, components], aggregate=False, robust=False)

    for c, (Ec, comp) in enumerate(zip(tf.unstack(E_comp, axis=-3), tf.unstack(components, axis=-2))):
        fig, axs = plt.subplots(1, 3, figsize=(8, 3))
        comp_mean = tf.reduce_mean(comp, axis=[-1, -2])
        im = axs[0].imshow(comp_mean[0], interpolation='none')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title('Component (nhood avg)', fontsize=7)
        fig.colorbar(im, ax=axs[0], fraction=0.02, pad=0.04)

        Ec_mean = tf.reduce_mean(Ec[..., 0], axis=[-1, -2])
        im = axs[1].imshow(Ec_mean[0], interpolation='none')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title('Component energy (non-robust)', fontsize=7)
        fig.colorbar(im, ax=axs[1], fraction=0.02, pad=0.04)

        is_rob_mean = tf.reduce_mean(tf.cast(Ec[..., 0] > N_rob_comp ** 2., tf.float32), axis=[-1, -2])
        im = axs[2].imshow(is_rob_mean[0], interpolation='none')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_title('In robust region (avg over nhood)', fontsize=7)
        fig.colorbar(im, ax=axs[2], fraction=0.02, pad=0.04)

        fig.suptitle(f'Iter: {itr}, component {c}')
        if plotfile is None:
            plt.show()
        else:
            plt.savefig(plotfile.replace('.png', f'_comp{c}.png'), bbox_inches='tight')
            plt.close('all')


def plot_filters(filters, coeffs=None, filter_std=None, filters_init=None, itr=None, plotfile=None, with_titles=False):

    with_coeffs = coeffs is not None
    nchan_in = filters.shape[2]
    filters_plot = tf.transpose(filters, [3, 0, 1, 2]).numpy()
    plot_init_filters = filters_init is not None
    if plot_init_filters:
        filters_init = tf.transpose(filters_init, [3, 0, 1, 2])[..., 0].numpy()
    else:
        filters_init = tf.zeros_like(filters_plot)

    if filter_std is not None:
        filter_std_mean = np.mean(np.mean(np.mean(filter_std, axis=0), 0), 0)
        plt.hist(filter_std_mean, bins=max(int(len(filter_std_mean) / 3), 1))
        plt.savefig(plotfile.replace('.p', f'_filters_std_hist.p'), bbox_inches='tight')
        plt.close('all')
    for i, f in enumerate(filters_plot):
        plt.close('all')
        nrow, ncol = int(np.ceil(np.sqrt(nchan_in))), int(np.ceil(nchan_in / np.ceil(np.sqrt(nchan_in))))
        fig, axs = plt.subplots(nrow, ncol)
        try:
            axs = [[a for a in ax] for ax in axs]
        except TypeError:
            try:
                axs = [[ax] for ax in axs]
            except TypeError:
                axs = [[axs]]
        for ax in axs:
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
        for c in range(nchan_in):
            row, col = c // ncol, c % ncol
            ax = axs[row][col]
            im = ax.imshow(f[..., c], interpolation='none')
            if with_titles:
                if filter_std is not None:
                    subtitle = f'Filt: {i}, Chan: {c}, Avgstd: {str(np.mean(filter_std[..., c, i]))[:5]}'
                else:
                    subtitle = f'Filt: {i}, Chan: {c}'
                ax.set_title(subtitle, fontsize=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            # cbar.ax.tick_params(labelsize=25)
        if with_titles:
            plt.suptitle(f'Iter: {itr}')
        if plotfile is not None:
            plt.savefig(plotfile.replace('.p', f'_filters{i}.p'), bbox_inches='tight')
            plt.close('all')
        else:
            plt.show()

        plt.close('all')
        if with_coeffs:
            plot_coeffs(coeffs[0], itr, i, plotfile)


def plot_dense_weights(weights, plotfile=None):
    imshow(weights)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig(plotfile, bbox_inches='tight')
    plt.close('all')


def plot_dense_input_output(ins, outs, plotfile=None, y_class=None):
    y_class = None if y_class is None else tf.argmax(y_class, axis=-1)
    fig, axs = plt.subplots(1, 2)
    for i, (a, x) in enumerate(zip(axs, (ins, outs))):
        a.set_xticks([])
        a.set_yticks([])
        if i == 1 and y_class is not None:
            a.scatter(y_class, range(len(y_class)), c='r', s=1.)
        im = a.imshow(x, interpolation='none')
        fig.colorbar(im, ax=a, fraction=0.04, pad=0.04)
    plt.tight_layout()
    plt.savefig(plotfile)
    plt.close('all')


def plot_coeffs(coeffs, itr=None, filter_id=None, plotfile=None):
    im = imshow(coeffs[..., filter_id])

    plot_title = ""
    if itr is not None:
        plot_title = f"Iter: {itr}"
    if filter_id is not None:
        plot_title += f" Coeffs for filter {filter_id}"
    if len(plot_title):
        plt.title(plot_title)
    plt.colorbar(fraction=0.02, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if plotfile is not None:
        plt.savefig(plotfile.replace('.p', f'_coeffs{filter_id}.p'), bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


def plot_coeffs_subplots(coeffs, itr=None, plotfile=None):
    n_filts = coeffs.shape[-1]
    nr = int(np.ceil(np.sqrt(n_filts)))
    nc = int(np.ceil(n_filts / nr))

    fig, axs = plt.subplots(nr, nc)
    if not isinstance(axs, (tuple, list, np.ndarray)):
        axs = [axs]
    if not isinstance(axs[0], (tuple, list, np.ndarray)):
        axs = [axs]
    for r, arow in enumerate(axs):
        for c, a in enumerate(arow):
            a.set_xticks([])
            a.set_yticks([])
            if c * nr + r > n_filts:
                a.set_axis_off()

    for f in range(n_filts):
        i, j = f // nr, f % nr
        im = axs[i][j].imshow(coeffs[..., f], interpolation='none')
        cbar = fig.colorbar(im, ax=axs[i][j], fraction=0.04, pad=0.04)

    if plotfile is not None:
        if itr is not None:
            pf = plotfile.replace('.p', f'_itr{itr}.p')
        else:
            pf = plotfile
        plt.savefig(pf, bbox_inches='tight')
        plt.close('all')
    else:
        plt.tight_layout()
        plt.show()
