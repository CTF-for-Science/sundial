import argparse
import yaml
import datetime

from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_prediction_timesteps, get_training_timesteps
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
import os

import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
    
def main(config_path: str) -> None:

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    # Model name
    model_name = f"{config['model']['name']}"
    
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True).to(device)

    if 'num_samples' in config['model']:
        num_samples = config['model']['num_samples']
    else:
        num_samples = 1

    if 'spatial_batch' in config['model']:
        spatial_batch = config['model']['spatial_batch']
    else:
        spatial_batch = 'all'

    # Define the name of the output folder for your batch
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize Visualization object
    viz = Visualization()

    # Get applicable visualizations for the dataset
    applicable_plots = get_applicable_plots(dataset_name)

    # Process each sub-dataset
    for pair_id in pair_ids:

        # Load sub-dataset - transpose is used since DMD requires data in (time, space) format
        train_data, init_data = load_dataset(dataset_name, pair_id)
        forecast_length = get_prediction_timesteps(dataset_name, pair_id).shape[0]

        # Load metadata
        train_timesteps = get_training_timesteps(dataset_name, pair_id)[0] # extract first element
        prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)

        if pair_id in [2, 4]:
            recon_ctx = 1000
            # Reconstruction
            print(f"> Reconstruction task, using {recon_ctx} context length")
            train_mat = train_data[0]
            train_mat = train_mat[0:recon_ctx,:]
            forecast_length = forecast_length - recon_ctx
        elif pair_id in [8, 9]:
            # Burn-in - Parametric Generalisation
            print(f"> Burn-in matrix of size {init_data.shape[0]}, using {forecast_length} forecast length")
            train_mat = init_data
            forecast_length = forecast_length - init_data.shape[0]
        else:
            # Standard prediction
            print(f"> Standard prediction task, using {forecast_length} forecast length")
            train_mat = train_data[0]

        # If GPU is too small, we can sequentially process the input data
        spatial_dim = train_mat.shape[-1]
        pred_data = np.zeros((spatial_dim, num_samples, forecast_length), dtype=np.float32)

        if spatial_batch == 'all':
            batch_size = spatial_dim
        else:
            batch_size = spatial_batch

        # Prediction in batches
        for i in tqdm(range(0, spatial_dim, batch_size), desc=f"Processing pair_id {pair_id} in batches"):

            _input_data = torch.tensor(train_mat[:, i : i + batch_size], dtype=torch.float32).to(device).T
            _tmp = model.generate(_input_data, max_new_tokens=forecast_length, num_samples=num_samples)
            pred_data[i : (i + batch_size)] = _tmp.cpu().numpy()

        if pair_id in [2, 4, 8, 9]:
            pred_mat = np.concatenate([train_mat, pred_data.mean(axis=1).T], axis=0)
        else:
            pred_mat = pred_data.mean(axis=1).T

        # Evaluate the model performance
        results = evaluate(dataset_name, pair_id, pred_mat)

        # Save results for this sub-dataset and get the path to the results directory
        results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_mat, results)

        # Append metrics to batch results
        # Convert metric values to plain Python floats for YAML serialization
        batch_results['pairs'].append({
            'pair_id': pair_id,
            'metrics': results
        })

        # Generate and save visualizations that are applicable to this dataset
        for plot_type in applicable_plots:
            fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
            viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

        print(' ')

    # Save aggregated batch results
    with open(results_directory.parent / 'batch_results.yaml', 'w') as f:
        yaml.dump(batch_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)