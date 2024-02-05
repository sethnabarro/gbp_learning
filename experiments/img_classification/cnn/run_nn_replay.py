"""Based on https://github.com/tensorflow/datasets/blob/master/docs/keras_example.ipynb"""
import numpy as np
import os
import shutil
import tensorflow as tf
import time

from experiments.img_classification.nn.utils import get_optim, get_data, get_fresh_model, FIFOBuffer

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

MASTER_RESDIR = 'results_w_replay_recent_trfrac_fix_rpb_bug'
BATCH_SIZE = 50

def main(tr_size, arch, epochs, lr, lr_replay,
         l2coeff, buffer_size, buffer_update_frac,
         n_inner_steps, seed=872, validation=False,
         n_outer_steps=1, n_repeats=1):
    ARCH = arch
    SEED = seed
    EPOCHS = epochs
    REPEATS = n_repeats
    VALIDATION = validation
    if VALIDATION:
        RESULTS_BASE_DIR = f'{MASTER_RESDIR}/validation/'
    else:
        RESULTS_BASE_DIR = f'{MASTER_RESDIR}/test/'
    ACTIVATION = 'LeakyReLU'
    OPTIM = 'adam'
    LR = lr
    LR_REPLAY = lr_replay
    BATCHSIZE = BATCH_SIZE
    BUFFER_SIZE = buffer_size
    BUFFER_UPDATE_FRAC = buffer_update_frac
    N_INNER_STEPS = n_inner_steps
    N_OUTER_STEPS = n_outer_steps
    L2_COEFF = l2coeff
    POOL_TYPE = 'max'
    TR_SIZE = tr_size
    RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f'{TR_SIZE}tr_{BUFFER_SIZE}buff',
                               f'{BUFFER_UPDATE_FRAC}buffupdfrac_'
                               f'{N_INNER_STEPS}ninstep_'
                               f'{f"{N_OUTER_STEPS}_" if N_OUTER_STEPS > 1 else ""}'
                               f'{ARCH}_{REPEATS}reps_'
                               f'{"validation_" if VALIDATION else ""}'
                               f'{TR_SIZE}trsize_'
                               f'{POOL_TYPE}pool_'
                               f'{BATCHSIZE}bs_'
                               f'{SEED}seed_'
                               f'{f"{L2_COEFF}l2_" if L2_COEFF > 0. else ""}'
                               f'{ACTIVATION.lower()}_{LR}lr_{LR_REPLAY}lr_{OPTIM}'.replace('.', '-'))

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
        opt_replay = get_optim(name=OPTIM, lr=LR_REPLAY)

        buffer = FIFOBuffer(n_examples=BUFFER_SIZE,
                            x_dims=ds_info.features['image'].shape,
                            y_dims=())

        def train_itr(x, y, optim=opt):
            with tf.GradientTape() as tape:
                tape.watch(x)
                pred = model_to_tr(x)
                loss = tf.reduce_mean(loss_fn(y, pred)) + model_to_tr.losses
            grad = tape.gradient(loss, model_to_tr.trainable_variables)
            optim.apply_gradients(zip(grad, model_to_tr.trainable_variables))
            return loss

        def replay_itr(buff):
            if (BATCHSIZE >= buff.counter) or (BATCHSIZE >= buff.size):
                # Only one batch
                x_rep, y_rep = buff.arrays
                for _ in range(N_INNER_STEPS):
                    train_itr(x_rep, y_rep, optim=opt_replay)
            else:
                buff_ds = buff.as_batched_dataset(BATCHSIZE)
                for step, (x_rep, y_rep) in enumerate(buff_ds):
                    if step >= N_INNER_STEPS:
                        break
                    train_itr(x_rep, y_rep, optim=opt_replay)

        def update_buffer(xtrain, ytrain, buff):
            n_upd = int(BUFFER_UPDATE_FRAC * xtrain.shape[0])
            x_for_upd = xtrain[:n_upd]
            y_for_upd = ytrain[:n_upd]
            buff.update_recent(x_for_upd, y_for_upd)

        for e in range(EPOCHS):
            loss_sum = 0.
            count = 0
            for xtr, ytr in train_ds:
                for _ in range(N_OUTER_STEPS - 1):
                    train_itr(xtr, ytr)
                loss_sum += train_itr(xtr, ytr)
                if (N_INNER_STEPS > 0 and
                        BUFFER_UPDATE_FRAC > 0. and
                        BUFFER_SIZE > 0):  # No point updating buffer if don't use it
                    update_buffer(xtr, ytr, buffer)
                    replay_itr(buffer)
                count += xtr.shape[0]
            print(f'\t\tEpoch: {e}, loss is {loss_sum / tf.cast(count, tf.float32)}')

        return model_to_tr

    results = np.empty(shape=(0, 3))
    for rep in range(REPEATS):
        print(f'Repeat {rep + 1} of {REPEATS}')
        model = get_fresh_model(arch_name=ARCH,
                                activation=ACTIVATION,
                                pool_type=POOL_TYPE,
                                l2_coeff=L2_COEFF,
                                seed=SEED + rep)
        ds_train, ds_test, ds_info = get_data(tr_size=TR_SIZE,
                                              validation=VALIDATION,
                                              shuffle_seed=SEED + rep,
                                              fashion=False)

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

        print(f'\t\tTest result: {[rep, acc.numpy(), ll.numpy()]}')
        results = np.concatenate([results, [[rep, acc, ll]]], axis=0)
        np.savetxt(os.path.join(RESULTS_DIR, "results.csv"), results, delimiter=",")
    return acc, RESULTS_DIR


if __name__ == '__main__':
    validation = False
    nonval_frac = 0.85
    train_sizes = [50,
                   100,
                   200,
                   400,
                   800,
                   3200,
                   12800,
                   int(60000 * nonval_frac) if validation else 60000][::-1]
    buffer_size_fracs = [0.003, 0.01, 0.03, 0.06, 0.1]

    for train_size in train_sizes:
        for buff_size_frac in buffer_size_fracs:
            if buff_size_frac * train_size < 1. and buff_size_frac > 0.:
                continue
            buff_size = int(np.ceil(buff_size_frac * train_size))

            acc_best = 0.
            val_or_test = 'validation' if validation else 'test'
            if os.path.exists(f'{MASTER_RESDIR}/{val_or_test}/{train_size}tr_{buff_size}buff'):
                if os.path.exists(f'{MASTER_RESDIR}/{val_or_test}/{train_size}tr_{buff_size}buff/best_config.txt'):
                    with open(f'{MASTER_RESDIR}/{val_or_test}/{train_size}tr_{buff_size}buff/best_config.txt', 'r') as resfile:
                        acc_best = float(resfile.read().split('New best acc: ')[-1].split('\n')[0])
            else:
                os.mkdir(f'{MASTER_RESDIR}/{val_or_test}/{train_size}tr_{buff_size}buff')
            # for l2 in [0., 1e-3]:

            for n_out_steps in [1, 3, 10, 30]:
                for buff_upd_frac in [0., 0.03, 0.1, 0.3, 0.5]:
                    if int(np.ceil(buff_upd_frac * BATCH_SIZE)) > buff_size and buff_size > 0:
                        # Size of each update would be bigger than buffer itself
                        continue
                    for n_step_inner in [0, 1, 3, 10, 30]: #([0] if buff_size == 0 else [0, 1, 3, 10]): #[0, 1, 3, 10, 20]:
                        for l_r in [5e-2, 5e-3, 5e-4]:
                            for l_r_rep in [5e-3, 5e-4, 5e-5]: #([5e-3] if buff_size == 0 else [5e-3, 5e-4]): #[5e-3, 5e-4, 5e-5]:
                                tbef = time.time()
                                acc_conf, resdir_conf = \
                                    main(tr_size=train_size,
                                         l2coeff=0.,
                                         lr=l_r,
                                         arch='conv_pool_dense_k5',
                                         epochs=1,
                                         seed=np.random.randint(np.iinfo(np.int32).max),
                                         buffer_size=buff_size,
                                         buffer_update_frac=buff_upd_frac,
                                         lr_replay=l_r_rep,
                                         n_inner_steps=n_step_inner,
                                         n_outer_steps=n_out_steps,
                                         validation=validation,
                                         n_repeats=1 if validation else 5)
                                taft = time.time()
                                print(f'Took {taft - tbef}')
                                if acc_conf > acc_best:
                                    best_conf_str = \
                                        f'Best conf so far for train size {train_size}, replay buffer size {buff_size}:'\
                                        f'\n\tLR: {l_r}\n\tLR replay: {l_r_rep}\n\t'\
                                        f'Buffer update prob: {buff_upd_frac}\n\tNum inner steps: {n_step_inner}\n\t'\
                                        f'Num outer steps: {n_out_steps}.'\
                                        f'\nNew best acc: {acc_conf}\nResults dir: {resdir_conf}\n'
                                    print(best_conf_str)
                                    acc_best = acc_conf
                                    with open(f'{MASTER_RESDIR}/{val_or_test}/{train_size}tr_{buff_size}buff/best_config.txt', 'a') as best_conf_file:
                                        best_conf_file.write(best_conf_str)

