import torch
from transformers import AutoModelForCausalLM
from ctf4science.data_module import load_dataset, parse_pair_ids, get_applicable_plots, get_prediction_timesteps, get_training_timesteps, load_validation_dataset, get_validation_prediction_timesteps
from ctf4science.eval_module import evaluate, save_results

import pickle
import os
import time
import numpy as np
from tqdm import tqdm

import argparse

def main(args=None):

    model_name = 'sundial'
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True).to(device)

    # Pair ids 2, 4: reconstruction
    # Pair ids 1, 3, 5-7: forecast
    # Pair ids 8, 9: burn-in
    dataset_name = args.dataset
    num_samples = args.num_samples
    batch_size = args.spatial_batch

    path_res = f"{dataset_name}/"
    os.makedirs(path_res, exist_ok=True)

    execution_time = time.time()

    for pair_id in range(1,10):

        print(f"Processing pair_id: {pair_id}")

        train_data, init_data = load_dataset(dataset_name, pair_id=pair_id)
        forecast_length = get_prediction_timesteps(dataset_name, pair_id).shape[0]


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

        for i in tqdm(range(0, spatial_dim, batch_size), desc=f"Processing pair_id {pair_id} in batches"):

            _input_data = torch.tensor(train_mat[:, i : i + batch_size], dtype=torch.float32).to(device).T
            _tmp = model.generate(_input_data, max_new_tokens=forecast_length, num_samples=num_samples)
            pred_data[i : (i + batch_size)] = _tmp.cpu().numpy()

        if pair_id in [2, 4, 8, 9]:
            pred_mat = np.concatenate([train_mat, pred_data.mean(axis=1).T], axis=0)
        else:
            pred_mat = pred_data.mean(axis=1).T

        # Evaluate the performance (mean prediction over samples)
        results = evaluate(dataset_name, pair_id, pred_mat)

        # Save results
        print(f"> Prediction matrix shape: {pred_mat.shape}")
        print(f"> Results: {results}")

        pickle.dump({
            'model_name': model_name,
            'dataset_name': dataset_name,
            'pair_id': pair_id,
            'pred_mat': pred_mat,
            'pred_shape': pred_data.shape,
            'results': results
        }, open(f"{dataset_name}/pair_{pair_id}_results.pkl", "wb"))

        print(' ')

    execution_time = time.time() - execution_time

    # Convert to HH:MM:SS format
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)

    print(f"> Total execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, required=True, help="Dataset to run (ODE_Lorenz or PDE_KS)")
    parser.add_argument('--spatial_batch', type=int, default=100, help="Number of 'spatial' batches to run - if the GPU is too small, we can sequentially process the input data")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples to generate for each prediction - CI of the foundation model")
    args = parser.parse_args()

    main(args)