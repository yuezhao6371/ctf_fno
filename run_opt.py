import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from ctf4science.data_module import load_validation_dataset, get_validation_prediction_timesteps, parse_pair_ids
from ctf4science.eval_module import evaluate_custom
from fno import FNO

# Delete results directory - used for storing batch_results
file_dir = Path(__file__).parent

def main(config_path: str) -> None:
    """
    Main function to run the FNO model on specified sub-datasets for hyperparameter optimization.

    Loads configuration, parses pair_ids, initializes the model, generates predictions,
    evaluates them, and saves results for each sub-dataset under a batch identifier.

    The evaluation function evaluates on validation data obtained from training data.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset name and get list of sub-dataset train/test pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])

    model_name = f"{config['model']['name']}"

    # batch_id is from optimize_parameters.py
    batch_id = f"hyper_opt_{config['model']['batch_id']}"
 
    # Initialize batch results dictionary for summary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Process each sub-dataset
    for pair_id in pair_ids:
        try:
            # Generate training and validation splits (and burn-in matrix when applicable) 
            train_split = config['model']['train_ratio']
            train_data, val_data, init_data = load_validation_dataset(dataset_name, pair_id, train_split, transpose=False)
            
            # Concatenate training data arrays along axis 0 (time dimension)
            if isinstance(train_data, list):
                train_data = np.concatenate(train_data, axis=0)
            
            # Load meta-data
            prediction_timesteps = get_validation_prediction_timesteps(dataset_name, pair_id, train_split)

            # Initialize model
            model = FNO(config, train_data, init_data, prediction_timesteps.shape[0], pair_id)

            # Train model
            model.train()

            # Generate predictions
            pred_data = model.predict()

            # Create fallback metrics in case evaluation fails
            fallback_metrics = {
                'short_time': float(-999),
                'long_time': float(-999),
                'reconstruction': float(-999)
            }
            
            # Try to evaluate predictions
            try:
                # Evaluate predictions using default metrics
                results = evaluate_custom(dataset_name, pair_id, val_data, pred_data)
            except Exception as e:
                print(f"Warning: Evaluation failed for pair {pair_id}: {str(e)}")
                results = fallback_metrics

            # Append metrics to batch results
            batch_results['pairs'].append({
                'pair_id': pair_id,
                'metrics': results
            })

        except Exception as e:
            print(f"Error processing pair {pair_id}: {str(e)}")
            continue

    # Save results
    results_path = file_dir / f"results_{config['model']['batch_id']}.yaml"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        yaml.dump(batch_results, f, default_flow_style=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config) 