# gbp-learning-anon
Code accompanying the paper
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
- `experiments/mnist/lm/run_lm.py`
  - Set variables `l2coeff`, `lr`, `epochs` as per the Appendix.
- `experiments/mnist/run_mnist_sample_efficiency.sh`
  - Run for each training set size by setting the `n_tr_data` variable

### Layerwise Asynchronous Training
- Run `experiments/mnist/run_mnist_synch.sh` and `experiments/mnist/run_mnist_asynch.sh`