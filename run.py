"""
Main script for running the FNO model on specified datasets.

This script handles:
- Loading the configuration file
- Processing one or multiple sub-datasets
- Training the FNO model
- Generating predictions
- Evaluating results
- Saving outputs and visualizations
"""

import argparse
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from ctf4science.data_module import (
    load_dataset,
    parse_pair_ids,
    get_applicable_plots,
    get_prediction_timesteps,
    get_training_timesteps
)
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from fno import FNO

# Set PyTorch memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"fno_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    
    return log_file


def main(config_path: str, log_level=logging.INFO):
    # Set up logging
    log_file = setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting FNO run with config: {config_path}")
    logger.info(f"Log file: {log_file}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract dataset name and parse pair_ids
    dataset_name = config['dataset']['name']
    pair_ids = parse_pair_ids(config['dataset'])
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Pair IDs: {pair_ids}")

    model_name = "FNO"
    # Generate a unique batch_id for this run
    batch_id = f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Batch ID: {batch_id}")

    # Initialize batch results dictionary
    batch_results = {
        'batch_id': batch_id,
        'model': model_name,
        'dataset': dataset_name,
        'pairs': []
    }

    # Initialize visualization object
    viz = Visualization()
    applicable_plots = get_applicable_plots(dataset_name)
    logger.info(f"Applicable plots: {applicable_plots}")

    # Process each sub-dataset
    for pair_id in pair_ids:
        try:
            logger.info(f"Processing pair {pair_id}")
            # Load sub-dataset
            train_data, init_data = load_dataset(dataset_name, pair_id)
            logger.debug(f"Training data type: {type(train_data)}")
            if isinstance(train_data, list):
                logger.debug(f"Training data list length: {len(train_data)}")
                for i, data in enumerate(train_data):
                    logger.debug(f"Training data[{i}] shape: {data.shape if hasattr(data, 'shape') else None}")
                # Handle pairs 8 and 9
                if pair_id in [8, 9]:
                    # For pairs 8 and 9, concatenate along axis 0 (time dimension)
                    train_data = np.concatenate(train_data, axis=0)
                else:
                    # For other pairs, concatenate along axis 1 (columns)
                    train_data = np.concatenate(train_data, axis=1)
                logger.debug(f"Concatenated training data shape: {train_data.shape}")
            else:
                logger.debug(f"Training data shape: {train_data.shape if train_data is not None else None}")
            logger.debug(f"Initialization data shape: {init_data.shape if init_data is not None else None}")

            # Load metadata (to provide forecast length)
            prediction_timesteps = get_prediction_timesteps(dataset_name, pair_id)
            prediction_horizon_steps = prediction_timesteps.shape[0]
            logger.debug(f"Prediction horizon steps: {prediction_horizon_steps}")

            # Initialize the model with the config and train_data
            model = FNO(config, train_data, init_data, prediction_horizon_steps, pair_id)
                    
            # Train the model
            logger.info(f"Training model for pair {pair_id}")
            model.train()
            
            # Generate predictions
            logger.info(f"Generating predictions for pair {pair_id}")
            pred_data = model.predict()
            logger.debug(f"Prediction data shape: {pred_data.shape}")

            # Evaluate predictions using default metrics
            logger.info(f"Evaluating predictions for pair {pair_id}")
            results = evaluate(dataset_name, pair_id, pred_data)
            
            # Print evaluation results
            logger.info(f"\nEvaluation Results for Pair {pair_id}:")
            logger.info("=" * 50)
            for metric_name, metric_value in results.items():
                logger.info(f"{metric_name}: {metric_value:.6f}")
            logger.info("=" * 50 + "\n")

            # Save results for this sub-dataset and get the path to the results directory
            results_directory = save_results(dataset_name, model_name, batch_id, pair_id, config, pred_data, results)
            logger.info(f"Results saved to: {results_directory}")

            # Append metrics to batch results
            batch_results['pairs'].append({
                'pair_id': pair_id,
                'metrics': results
            })

            # Generate and save visualizations that are applicable to this dataset
            for plot_type in applicable_plots:
                logger.info(f"Generating {plot_type} plot for pair {pair_id}")
                fig = viz.plot_from_batch(dataset_name, pair_id, results_directory, plot_type=plot_type)
                viz.save_figure_results(fig, dataset_name, model_name, batch_id, pair_id, plot_type, results_directory)

        except Exception as e:
            logger.error(f"Error processing pair {pair_id}: {str(e)}", exc_info=True)
            continue

    # Print final summary of all pairs
    logger.info("\nFinal Summary of All Pairs:")
    logger.info("=" * 50)
    for pair_result in batch_results['pairs']:
        pair_id = pair_result['pair_id']
        metrics = pair_result['metrics']
        logger.info(f"\nPair {pair_id}:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.6f}")
    logger.info("=" * 50)

    # Save aggregated batch results
    batch_results_path = results_directory.parent / 'batch_results.yaml'
    with open(batch_results_path, 'w') as f:
        yaml.dump(batch_results, f)
    logger.info(f"Batch results saved to: {batch_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the configuration file")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    
    # Set log level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    main(args.config, log_level) 