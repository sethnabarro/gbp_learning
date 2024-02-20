# gbp-learning-anon
Code accompanying GBP Learning paper:
> *Learning in Deep Factor Graphs with Gaussian Belief Propagation*

## Dependencies
Install from the `requirements.txt` file.

## Reproducing Results
After setting up your python environment, set the `ENV_DIR` variable in the experiment scripts accordingly. The results for each experiment will be written to a subdirectory.
For example running `experiments/video/run_pairwise.sh` will create a folder `experiments/video/results/final_pairwise/` where the results will be stored.

### Figure 2b
Run script `experiments/xor/run.sh`.

### Figure 2c
Run script `experiments/regression/run.sh`.

### Figure 3
Run scripts 
- `experiments/video/run_five_layer_no_filtering.sh`
- `experiments/video/run_five_layer_w_filtering.sh`
- `experiments/video/run_single_layer_no_filtering.sh`
- `experiments/video/run_single_layer_with_filtering.sh`
- `experiments/video/run_pairwise.sh`

### Figure 4
Run scripts 
- For linear classifier baseline: `experiments/img_classification/mnist_lm/run_lm.py`
  - Hyperparameter tuning: `experiments/img_classification/mnist_lm/run_lm_validation.py`
- For CNN baseline: `experiments/img_classification/mnist_cnn/run_nn_replay.py`
  - Set variable `validation=True` during hyperparameter tuning.
- For GBP Learning: `experiments/img_classification/run_mnist_sample_efficiency.sh`
  - Run for each training set size by setting the `n_tr_data` variable

### Layerwise Asynchronous Training
- Run `experiments/img_classification/run_mnist_asynch.sh`.
  - Compare to results of `experiments/img_classification/run_mnist_sample_efficiency.sh` with `n_tr_data=60000` (full training set) as this is with synchronous forward/backward sweeps by default.

### Compare with Lucibello et al., 2022
- For MNIST, use results of `experiments/img_classification/run_mnist_sample_efficiency.sh` with `n_tr_data=60000` (full training set)
- For FashionMNIST: `experiments/img_classification/run_fmnist.sh`
- For CIFAR10: `experiments/img_classification/run_cifar10.sh`
