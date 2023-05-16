"""https://github.com/tensorflow/datasets/blob/master/docs/keras_example.ipynb"""
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.keras.regularizers import L2 as L2Reg
import tensorflow_datasets as tfds

ARCH = 'cnn'
EPOCHS = 2
REPEATS = 6
NO_TRAIN_CROSSTALK = False
NO_TEST_CROSSTALK = False
VALIDATION = False
RESULTS_BASE_DIR = '../nn/results/'
if VALIDATION:
    RESULTS_BASE_DIR += 'validation/validation_15percent'
else:
    RESULTS_BASE_DIR += 'test/'
REINIT_OPTIM = False
ACTIVATION = 'LeakyReLU'
OPTIM = 'adam'
LR = 0.000015
L2_COEFF = 0.2
POOL_TYPE = 'avg'
N_EXAMPLES_PER_CLASS_TR = None
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR,
                           f'{ARCH}_{EPOCHS}epochs_{REPEATS}reps_'
                           f'{"validation_" if VALIDATION else ""}'
                           f'{"no" if NO_TRAIN_CROSSTALK else ""}trxtalk_'
                           f'{"no" if NO_TEST_CROSSTALK else ""}textalk_'
                           f'{"" if REINIT_OPTIM else "no"}reinitoptim_'
                           f'{N_EXAMPLES_PER_CLASS_TR}xperclasstr_'
                           f'{POOL_TYPE}pool_'
                           f'{f"{L2_COEFF}l2_" if L2_COEFF > 0. else ""}'
                           f'{ACTIVATION.lower()}_{LR}lr_{OPTIM}')

if os.path.exists(RESULTS_DIR):
    shutil.rmtree(RESULTS_DIR)
os.mkdir(RESULTS_DIR)

code_filepath = os.path.abspath(__file__)
print(code_filepath)
shutil.copy(code_filepath, os.path.join(RESULTS_DIR, os.path.basename(__file__).replace('.py', '_bak.py')))


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def get_fresh_model():
    if ARCH == 'mlp':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(256, activation=ACTIVATION),
            tf.keras.layers.Dense(256, activation=ACTIVATION),
            tf.keras.layers.Dense(10)
        ])
    elif ARCH == 'wide_mlp':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(400, activation=ACTIVATION),
            tf.keras.layers.Dense(400, activation=ACTIVATION),
            tf.keras.layers.Dense(10)
        ])
    elif ARCH == 'smaller_mlp':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(256, activation=ACTIVATION),
            tf.keras.layers.Dense(10)
        ])
    elif ARCH == 'bigger_mlp':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(256, activation=ACTIVATION),
            tf.keras.layers.Dense(256, activation=ACTIVATION),
            tf.keras.layers.Dense(256, activation=ACTIVATION),
            tf.keras.layers.Dense(10)
        ])
    elif ARCH == 'cnn':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8, kernel_size=5, activation=ACTIVATION, kernel_regularizer=L2Reg(l2=L2_COEFF)),
            {'max': tf.keras.layers.MaxPool2D(2), 'avg': tf.keras.layers.AvgPool2D(2)}[POOL_TYPE],
            tf.keras.layers.Conv2D(16, kernel_size=3, activation=ACTIVATION, kernel_regularizer=L2Reg(l2=L2_COEFF)),
            {'max': tf.keras.layers.MaxPool2D(2), 'avg': tf.keras.layers.AvgPool2D(2)}[POOL_TYPE],
            tf.keras.layers.Conv2D(32, kernel_size=3, activation=ACTIVATION, kernel_regularizer=L2Reg(l2=L2_COEFF)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, kernel_regularizer=L2Reg(l2=L2_COEFF))
        ])

    model.build(input_shape=(None, 28, 28, 1))
    return model


def get_optim():
    if OPTIM == 'adam':
        opt = tf.keras.optimizers.Adam(LR)
    elif OPTIM == 'sgd':
        opt = tf.keras.optimizers.SGD(LR)
    return opt


def main():


    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(0.001),
    #     loss=_loss,
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    # )

    tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    results = np.empty(shape=(0, 5))
    for rep in range(REPEATS):
        print(f'Repeat {rep + 1} of {REPEATS}')
        model = get_fresh_model()
        for taskid, task in enumerate(tasks):
            print(f'\tTask is {task}')
            (ds_train, ds_test), ds_info = tfds.load(
                'mnist',
                split=['train[:85%]', 'train[85%:]'] if VALIDATION else ['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
            )

            ds_train = ds_train.map(
                normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            ds_train = ds_train.cache()

            ds_test = ds_test.map(
                normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
            ds_test = ds_test.cache()

            ds_train_task = ds_train.filter(lambda _, y: y == task[0] or y == task[1])
            ds_train_task = ds_train_task.shuffle(ds_info.splits['train'].num_examples)
            ds_train_task = ds_train_task.batch(100)
            ds_train_task = ds_train_task.prefetch(tf.data.AUTOTUNE)

            def _loss(true, pred):
                if NO_TRAIN_CROSSTALK:
                    return tf.keras.losses.sparse_categorical_crossentropy(true - task[0],
                                                                           pred[..., task[0]:task[1] + 1],
                                                                           from_logits=True,)
                else:
                    return tf.keras.losses.sparse_categorical_crossentropy(true,
                                                                           pred,
                                                                           from_logits=True)

            if taskid > 0:
                if REINIT_OPTIM:
                    opt = get_optim()
            else:
                opt = get_optim()

            def train_itr(x, y):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    pred = model(x)
                    loss = tf.reduce_mean(_loss(y, pred))
                grad = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grad, model.trainable_variables))
                return loss

            for e in range(EPOCHS):
                loss_sum = 0.
                count = 0
                for xtr, ytr in ds_train_task:
                    loss_sum += train_itr(xtr, ytr)
                    count += xtr.shape[0]
                    if N_EXAMPLES_PER_CLASS_TR is not None:
                        if count > N_EXAMPLES_PER_CLASS_TR * 2:
                            break
                print(f'\t\tEpoch: {e}, loss is {loss_sum / tf.cast(count, tf.float32)}')

            print(tf.reduce_mean(model.layers[-1].weights[0], axis=0))

            for taskid_te, task_te in enumerate(tasks):
                ds_test_task = ds_test.filter(lambda _, y: y == task_te[0] or y == task_te[1])
                ds_test_task = ds_test_task.batch(100)

                n_corr = 0.
                nll = 0.
                n_te_data = 0
                for xte, yte in ds_test_task:
                    logits = model.predict(xte)
                    if NO_TEST_CROSSTALK:
                        class_pred = tf.argmax(logits[..., task_te[0]:task_te[1] + 1], axis=-1,) + task_te[0]
                        nll_elem = tf.keras.losses.sparse_categorical_crossentropy(yte - task_te[0],
                                                                                   logits[...,
                                                                                   task_te[0]:task_te[1] + 1],
                                                                                   from_logits=True)
                    else:
                        class_pred = tf.argmax(logits, axis=-1)
                        nll_elem = tf.keras.losses.sparse_categorical_crossentropy(yte,
                                                                                   logits,
                                                                                   from_logits=True)
                    n_corr += tf.reduce_sum(tf.cast(class_pred == yte, tf.float32))
                    nll += tf.reduce_sum(nll_elem)
                    n_te_data += xte.shape[0]
                acc = n_corr / tf.cast(n_te_data, tf.float32)
                ll = -nll / tf.cast(n_te_data, tf.float32)

                print(f'\t\tResult row: {[rep, taskid, taskid_te, acc.numpy(), ll.numpy()]}')
                results = np.concatenate([results, [[rep, taskid, taskid_te, acc, ll]]], axis=0)
                np.savetxt(os.path.join(RESULTS_DIR, "results.csv"), results, delimiter=",")

    mem_accs = []
    for t in range(len(tasks)):
        mem_accs += results[(results[:, 1] == t) & (results[:, 2] < t), -2].tolist()
    avg_acc = np.mean(mem_accs)
    np.savetxt(os.path.join(RESULTS_DIR, f'avg_mem_acc_{str(round(avg_acc, 7)).replace(".", "_")}.txt'), [])

    for j, m in enumerate(['acc', 'll']):
        fig, axs = plt.subplots(1, len(tasks))
        fig.set_size_inches([12, 3])
        for r in range(REPEATS):
            for i, t in enumerate(tasks):
                results_task = results[(results[:, 2] == i) & (results[:, 0] == r)]
                axs[i].plot(results_task[:, j + 3])
                axs[i].scatter(results_task[:, 1], results_task[:, j + 3])
                if m == 'acc' and NO_TEST_CROSSTALK:
                    axs[i].axhline(0.5, -1, 5, linestyle='--', color='k')
                axs[i].set_xticks(list(range(5)))
                axs[i].set_xlim([-1, 5])
                if m == 'acc' and NO_TEST_CROSSTALK:
                    axs[i].set_ylim([0.45, 1.05])
                else:
                    axs[i].set_ylim([-0.05, 1.05])
        plt.savefig(f'{RESULTS_DIR}/nn_baseline_{m}.png', bbox_inches='tight')
        plt.close('all')

    for j, m in enumerate(['acc', 'll']):
        fig, axs = plt.subplots(1, len(tasks))
        fig.set_size_inches([12, 3])
        for i, t in enumerate(tasks):
            results_test_task = results[results[:, 2] == i]
            results_mean = np.empty((0, 2))
            results_std = np.empty((0, 2))
            for tr_task_id in np.unique(results_test_task[:, 1],):  # Average over repeats
                res_mean = np.mean(results_test_task[results_test_task[:, 1] == tr_task_id], axis=0)[[1, j + 3]]
                res_std = np.std(results_test_task[results_test_task[:, 1] == tr_task_id], axis=0)[[1, j + 3]]
                results_mean = np.concatenate([results_mean, [res_mean]], axis=0)
                results_std = np.concatenate([results_std, [res_std]], axis=0)

            l, = axs[i].plot(results_mean[:, 1])
            axs[i].scatter(results_mean[:, 0], results_mean[:, 1], c=l.get_color())
            axs[i].fill_between(results_mean[:, 0],
                                results_mean[:, 1] - results_std[:, 1],
                                results_mean[:, 1] + results_std[:, 1],
                                alpha=0.3,
                                color=l.get_color()
                                )
            if m == 'acc' and NO_TEST_CROSSTALK:
                axs[i].axhline(0.5, -1, 5, linestyle='--', color='k')
            axs[i].set_xticks(list(range(5)))
            axs[i].set_xlim([-1, 5])
            if m == 'acc' and NO_TEST_CROSSTALK:
                axs[i].set_ylim([0.45, 1.05])
            else:
                axs[i].set_ylim([-0.05, 1.05])

        plt.savefig(f'{RESULTS_DIR}/nn_baseline_{m}_ci.png', bbox_inches='tight')
        plt.close('all')

        task_diff = results[:, 1] - results[:, 2]
        results = np.concatenate([results, task_diff[:, None]], axis=-1)
        td_to_acc = {}
        td_to_acc_std = {}
        for td in np.unique(task_diff):
            if td >= 0:
                td_to_acc[td] = np.mean(results[results[:, -1] == td, 3 + j], axis=0)
                td_to_acc_std[td] = np.std(results[results[:, -1] == td, 3 + j], axis=0)

        td_to_acc = np.array(list(zip(td_to_acc.keys(), td_to_acc.values())))
        td_to_acc_std = np.array(list(zip(td_to_acc_std.keys(), td_to_acc_std.values())))
        l, = plt.plot(td_to_acc[:, 0], td_to_acc[:, 1])
        plt.fill_between(td_to_acc[:, 0],
                         td_to_acc[:, 1] - td_to_acc_std[:, 1],
                         td_to_acc[:, 1] + td_to_acc_std[:, 1],
                         alpha=0.3,
                         color=l.get_color()
                         )
        plt.savefig(f'{RESULTS_DIR}/avg_memorisation_{m}_ci.png', bbox_inches='tight')
        plt.close('all')


if __name__ == '__main__':
    main()
