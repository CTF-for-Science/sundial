# Examples

`quickstart_zero_shot_generation.ipynb` is the Jupyter notebook that demonstrates how to use the Sundial model for zero-shot generation from the original repository.

`run.py` and `run.ipynb` are scripts that allow you to run Sundial on different datasets by specifying configuration files.

## How to Run

Both the jupyter notebook and the python script can be used to run the model.

Running the .py file:

```bash
python run.py --dataset ODE_Lorenz --spatial_batch 3 --num_samples 1
```

where:
- `--dataset` specifies the dataset to run (e.g., ODE_Lorenz or PDE_KS).
- `--spatial_batch` specifies the number of 'spatial' batches to run (required if the GPU is too small).
- `--num_samples` defines the number of samples to generate for each prediction (for confidence intervals of the foundation model).
