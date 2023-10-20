"""Based on https://github.com/tensorflow/datasets/blob/master/docs/keras_example.ipynb"""
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf

from experiments.mnist.nn.utils import get_optim, get_data, get_fresh_model


def main(tr_size, arch, epochs, lr, l2coeff, seed=872):
    ARCH = arch
    SEED = seed
    EPOCHS = epochs
    REPEATS = 1
    VALIDATION = True
    RESULTS_BASE_DIR = 'results/'
    if VALIDATION:
        RESULTS_BASE_DIR += 'validation/'
    else:
        RESULTS_BASE_DIR += 'test/'
    REINIT_OPTIM = False
    ACTIVATION = 'LeakyReLU'
    OPTIM = 'adam'
    LR = lr
    BATCHSIZE = 50
    L2_COEFF = l2coeff
    POOL_TYPE = 'max'
    MASK_FILL = 'uniform'
    NOISE_FRAC = 0.
    TR_SIZE = tr_size
    TEST_MASK_LENGTHS = [0]
    mask_str = '' if TEST_MASK_LENGTHS[0] == 0 and len(TEST_MASK_LENGTHS) == 1 else f'{MASK_FILL}maskfill_'
    RESULTS_DIR = os.path.join(RESULTS_BASE_DIR,
                               f'{mask_str}'
                               f'{ARCH}_{EPOCHS}epochs_{REPEATS}reps_'
                               f'{"validation_" if VALIDATION else ""}'
                               f'{"" if REINIT_OPTIM else "no"}reinitoptim_'
                               f'{TR_SIZE}trsize_'
                               f'{POOL_TYPE}pool_'
                               f'{BATCHSIZE}bs_'
                               f'{SEED}seed_'
                               f'{f"{L2_COEFF}l2_" if L2_COEFF > 0. else ""}'
                               f'{ACTIVATION.lower()}_{LR}lr_{OPTIM}'.replace('.', '-'))

    IMG_H = 28
    IMG_W = 28

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)

    code_filepath = os.path.abspath(__file__)
    shutil.copy(code_filepath, os.path.join(RESULTS_DIR, os.path.basename(__file__).replace('.py', '_bak.py')))

    def train(model_to_tr, train_ds):
        train_ds = train_ds.shuffle(ds_info.splits['train'].num_examples)
        train_ds = train_ds.batch(BATCHSIZE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt = get_optim(name=OPTIM, lr=LR)

        def train_itr(x, y):
            with tf.GradientTape() as tape:
                tape.watch(x)
                pred = model_to_tr(x)
                loss = tf.reduce_mean(loss_fn(y, pred)) + model_to_tr.losses
            grad = tape.gradient(loss, model_to_tr.trainable_variables)
            opt.apply_gradients(zip(grad, model_to_tr.trainable_variables))
            return loss

        for e in range(EPOCHS):
            loss_sum = 0.
            count = 0
            for xtr, ytr in train_ds:
                loss_sum += train_itr(xtr, ytr)
                count += xtr.shape[0]
            print(f'\t\tEpoch: {e}, loss is {loss_sum / tf.cast(count, tf.float32)}')

        return model_to_tr

    results = np.empty(shape=(0, 4))
    for rep in range(REPEATS):
        print(f'Repeat {rep + 1} of {REPEATS}')
        model = get_fresh_model(arch_name=ARCH,
                                activation=ACTIVATION,
                                pool_type=POOL_TYPE,
                                l2_coeff=L2_COEFF)
        ds_train, ds_test, ds_info = get_data(tr_size=TR_SIZE,
                                              validation=VALIDATION,
                                              shuffle_seed=SEED)

        model = train(model_to_tr=model, train_ds=ds_train)

        ds_test = ds_test.batch(BATCHSIZE)

        n_corr = 0.
        nll = 0.
        n_te_data = 0

        for masklen in TEST_MASK_LENGTHS:
            mask_start_min = 4
            mask_start_max = 24 - masklen
            for b, (xte, yte) in enumerate(ds_test):
                mask_idx = tf.random.uniform(shape=(BATCHSIZE, 2), minval=mask_start_min, maxval=mask_start_max, dtype=tf.int32)
                w_cond = tf.logical_and(mask_idx[:, None, 1] <= tf.range(IMG_W)[None],  tf.range(IMG_W)[None] < mask_idx[:, None, 1] + masklen)
                h_cond = tf.logical_and(mask_idx[:, None, 0] <= tf.range(IMG_H)[None], tf.range(IMG_H)[None] < mask_idx[:, None, 0] + masklen)
                mask_cond = tf.logical_and(w_cond[:, None], h_cond[:, :, None])[..., None]
                if MASK_FILL is None:
                    xte_mask = xte
                else:
                    if MASK_FILL == 'zeros':
                        mask_fill = 0.
                    elif MASK_FILL == 'uniform':
                        mask_fill = tf.random.uniform(shape=xte.shape,
                                                      minval=0.,
                                                      maxval=1.)
                    xte_mask = tf.where(mask_cond, mask_fill, xte)

                if NOISE_FRAC is None:
                    xte_mask_noise = xte_mask
                else:
                    noise = tf.random.uniform(shape=xte.shape,
                                              minval=0.,
                                              maxval=1.)
                    noise_pix = tf.random.uniform(shape=xte.shape,
                                                  minval=0.,
                                                  maxval=1.) < NOISE_FRAC
                    xte_mask_noise = tf.where(noise_pix, noise, xte_mask)
                # xte_mask_noise += tf.random.normal(shape=xte.shape, stddev=0.5)
                if b == 0:
                    plt.imshow(xte_mask_noise[0])
                    plt.show()
                # if b == 0:
                #     for f in range(5):
                #         plt.imshow(xte_mask[f, ..., 0])
                #         plt.show()
                logits = model.predict(xte_mask_noise)
                class_pred = tf.argmax(logits, axis=-1)
                nll_elem = tf.keras.losses.sparse_categorical_crossentropy(yte,
                                                                           logits,
                                                                           from_logits=True)
                n_corr += tf.reduce_sum(tf.cast(class_pred == yte, tf.float32))
                nll += tf.reduce_sum(nll_elem)
                n_te_data += xte.shape[0]
            acc = n_corr / tf.cast(n_te_data, tf.float32)
            ll = -nll / tf.cast(n_te_data, tf.float32)

            print(f'\t\tResult row: {[rep, masklen, acc.numpy(), ll.numpy()]}')
            results = np.concatenate([results, [[rep, masklen, acc, ll]]], axis=0)
            np.savetxt(os.path.join(RESULTS_DIR, "results.csv"), results, delimiter=",")
    return acc


if __name__ == '__main__':
    main(tr_size=None,
         l2coeff=0.,
         lr=0.003,
         arch='conv_pool_dense_k5',
         epochs=50,
         seed=944)
