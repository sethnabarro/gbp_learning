"""Based on https://github.com/tensorflow/datasets/blob/master/docs/keras_example.ipynb"""
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf

from experiments.img_classification.lm.utils import get_optim, get_data, get_model


def main(tr_size, epochs, lr, l2coeff, seed=872):
    SEED = seed
    EPOCHS = epochs
    REPEATS = 1
    VALIDATION = False
    RESULTS_BASE_DIR = 'results_lin/'
    if VALIDATION:
        RESULTS_BASE_DIR += 'validation/'
    else:
        RESULTS_BASE_DIR += 'test/'
    REINIT_OPTIM = False
    OPTIM = 'adam'
    LR = lr
    BATCHSIZE = 50
    L2_COEFF = l2coeff
    TR_SIZE = tr_size
    RESULTS_DIR = os.path.join(RESULTS_BASE_DIR,
                               f'{EPOCHS}epochs_{REPEATS}reps_'
                               f'{"validation_" if VALIDATION else ""}'
                               f'{"" if REINIT_OPTIM else "no"}reinitoptim_'
                               f'{TR_SIZE}trsize_'
                               f'{BATCHSIZE}bs_'
                               f'{SEED}seed_'
                               f'{f"{L2_COEFF}l2_" if L2_COEFF > 0. else ""}'
                               f'{LR}lr_{OPTIM}'.replace('.', '-'))


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
        model = get_model(l2_coeff=L2_COEFF)
        ds_train, ds_test, ds_info = get_data(tr_size=TR_SIZE,
                                              validation=VALIDATION,
                                              shuffle_seed=SEED)

        model = train(model_to_tr=model, train_ds=ds_train)

        ds_test = ds_test.batch(BATCHSIZE)

        n_corr = 0.
        nll = 0.
        n_te_data = 0
        for b, (xte, yte) in enumerate(ds_test):
            logits = model.predict(xte)
            class_pred = tf.argmax(logits, axis=-1)
            nll_elem = tf.keras.losses.sparse_categorical_crossentropy(yte,
                                                                       logits,
                                                                       from_logits=True)
            n_corr += tf.reduce_sum(tf.cast(class_pred == yte, tf.float32))
            nll += tf.reduce_sum(nll_elem)
            n_te_data += xte.shape[0]
        acc = n_corr / tf.cast(n_te_data, tf.float32)
        ll = -nll / tf.cast(n_te_data, tf.float32)

        print(f'\t\tResult row: {[rep, acc.numpy(), ll.numpy()]}')
        results = np.concatenate([results, [[rep, acc, ll]]], axis=0)
        np.savetxt(os.path.join(RESULTS_DIR, "results.csv"), results, delimiter=",")
    return acc


if __name__ == '__main__':
    main(tr_size=None,
         l2coeff=0.,
         lr=0.003,
         epochs=50,
         seed=944)
