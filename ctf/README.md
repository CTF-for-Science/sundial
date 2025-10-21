# Sundial for CTF4Science

This directory contains the evaluation code for the Sundial foundation model within the CTF4Science framework.


## Setup

To run the model, make sure to install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

Then install the ctf4science package from the git repository.

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
