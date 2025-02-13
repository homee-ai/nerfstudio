import os
import argparse
from pathlib import Path
import torch
import yaml
import logging
from typing import List, Dict, Optional
from datetime import datetime
from copy import deepcopy
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.eval_utils import eval_setup

def setup_logger(debug: bool = False) -> logging.Logger:
    """Set up logger with appropriate level and formatting."""
    logger = logging.getLogger('combine_chunks')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Create formatters and add it to the handlers
    basic_formatter = logging.Formatter('%(message)s')
    debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    ch.setFormatter(debug_formatter if debug else basic_formatter)
    logger.addHandler(ch)
    
    return logger

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def load_chunk_checkpoint(checkpoint_path: Path, device: str = "cuda", use_mask: bool = True, logger: Optional[logging.Logger] = None) -> Dict[str, torch.Tensor]:
    """Load gaussian parameters and other necessary fields from a checkpoint file."""
    if logger is None:
        logger = logging.getLogger('combine_chunks')
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print("===================")
    print("number of points in global visible mask")
    print(checkpoint['pipeline']['_model.global_visible_mask'].shape)
    # print number of points in means
    print("number of points in means")
    print(checkpoint['pipeline']['_model.gauss_params.means'].shape)
    print("===================")
    
    # Keep the original checkpoint structure but update pipeline state
    result = {
        'step': checkpoint.get('step', 0),
        'pipeline': {},  # Will extract gaussian params
        'optimizers': checkpoint.get('optimizers', {}),
        'schedulers': checkpoint.get('schedulers', {}),
        'frames': checkpoint.get('frames', []),
        'cameras': checkpoint.get('cameras', {}),
        'scalers': checkpoint.get('scalers', {}),  # Add scalers field
    }
    
    # Extract gaussian parameters from pipeline state
    pipeline_state = checkpoint['pipeline']
    
    # Get the global visible mask
    global_visible_mask = pipeline_state.get('_model.global_visible_mask', None)
    if global_visible_mask is not None and use_mask:
        # Filter out gaussians where mask is 0
        mask = global_visible_mask.bool()
        
        # Also filter optimizer states if they exist
        for opt_name in ['means', 'scales', 'quats', 'features_dc', 'features_rest', 'opacities']:
            if opt_name in checkpoint['optimizers'] and 0 in checkpoint['optimizers'][opt_name]['state']:
                state = checkpoint['optimizers'][opt_name]['state'][0]
                if 'exp_avg' in state:
                    state['exp_avg'] = state['exp_avg'][mask]
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'] = state['exp_avg_sq'][mask]
        
        result['pipeline'] = {
            **pipeline_state,
            '_model.gauss_params.means': pipeline_state['_model.gauss_params.means'][mask],
            '_model.gauss_params.scales': pipeline_state['_model.gauss_params.scales'][mask],
            '_model.gauss_params.quats': pipeline_state['_model.gauss_params.quats'][mask],
            '_model.gauss_params.features_dc': pipeline_state['_model.gauss_params.features_dc'][mask],
            '_model.gauss_params.features_rest': pipeline_state['_model.gauss_params.features_rest'][mask],
            '_model.gauss_params.opacities': pipeline_state['_model.gauss_params.opacities'][mask],
            '_model.global_visible_mask': torch.ones(mask.sum(), dtype=torch.bool, device=mask.device)
        }
    else:
        # If no mask or use_mask is False, keep all gaussians
        result['pipeline'] = {
            **pipeline_state,
            '_model.gauss_params.means': pipeline_state['_model.gauss_params.means'],
            '_model.gauss_params.scales': pipeline_state['_model.gauss_params.scales'],
            '_model.gauss_params.quats': pipeline_state['_model.gauss_params.quats'],
            '_model.gauss_params.features_dc': pipeline_state['_model.gauss_params.features_dc'],
            '_model.gauss_params.features_rest': pipeline_state['_model.gauss_params.features_rest'],
            '_model.gauss_params.opacities': pipeline_state['_model.gauss_params.opacities'],
            '_model.global_visible_mask': torch.zeros(pipeline_state['_model.gauss_params.means'].shape[0], dtype=torch.bool, device=device)
        }
    
    # Print number of gaussians in this chunk
    num_gaussians = result['pipeline']['_model.gauss_params.means'].shape[0]
    num_bil_grids = result['pipeline']['_model.bil_grids.grids'].shape[0] if '_model.bil_grids.grids' in result['pipeline'] else 0
    logger.info(f"  Number of gaussians: {num_gaussians}")
    logger.info(f"  Number of bilateral grids: {num_bil_grids}")
    
    return result

def combine_gaussian_models(chunk_params: List[Dict[str, torch.Tensor]], device: str = "cuda", logger: Optional[logging.Logger] = None) -> Dict:
    """Combine multiple chunks of checkpoints into one."""
    if logger is None:
        logger = logging.getLogger('combine_chunks')
        
    # Start with a copy of the first checkpoint's structure
    combined = deepcopy(chunk_params[0])
    
    # Combine gaussian parameters
    gauss_keys = [
        '_model.gauss_params.means',
        '_model.gauss_params.scales',
        '_model.gauss_params.quats',
        '_model.gauss_params.features_dc',
        '_model.gauss_params.features_rest',
        '_model.gauss_params.opacities'
    ]
    
    # Concatenate gaussian parameters from all chunks
    for key in gauss_keys:
        filtered_params = []
        for chunk_idx, chunk in enumerate(chunk_params):
            filtered_params.append(chunk['pipeline'][key])
        combined['pipeline'][key] = torch.cat(filtered_params, dim=0)
    
    # Remove global_visible_mask from combined checkpoint since we've filtered already
    if '_model.global_visible_mask' in combined['pipeline']:
        del combined['pipeline']['_model.global_visible_mask']
    
    # Handle bilateral grids separately - these don't need filtering
    if '_model.bil_grids.grids' in chunk_params[0]['pipeline']:
        combined['pipeline']['_model.bil_grids.grids'] = torch.cat(
            [p['pipeline']['_model.bil_grids.grids'] for p in chunk_params], 
            dim=0
        )
    
    # Handle camera optimizer parameters
    if '_model.camera_optimizer.pose_adjustment' in chunk_params[0]['pipeline']:
        combined['pipeline']['_model.camera_optimizer.pose_adjustment'] = torch.cat(
            [p['pipeline']['_model.camera_optimizer.pose_adjustment'] for p in chunk_params],
            dim=0
        )
    # Handle optimizer states
    optimizer_names = ['means', 'scales', 'quats', 'features_dc', 'features_rest', 'opacities']
    
    if '_model.bil_grids.grids' in chunk_params[0]['pipeline']:
        optimizer_names.append('bilateral_grid')
    if '_model.camera_optimizer.pose_adjustment' in chunk_params[0]['pipeline']:
        optimizer_names.append('camera_opt')
    for opt_name in optimizer_names:
        if opt_name in combined['optimizers']:
            # Keep the param_groups settings from the first chunk
            
            # Get the corresponding parameter shape for gaussian parameters
            if opt_name in ['means', 'scales', 'quats', 'features_dc', 'features_rest', 'opacities']:
                param_key = f'_model.gauss_params.{opt_name}'
                param_shape = combined['pipeline'][param_key].shape
            elif opt_name == 'bilateral_grid':
                param_shape = combined['pipeline']['_model.bil_grids.grids'].shape
            elif opt_name == 'camera_opt':
                param_shape = combined['pipeline']['_model.camera_optimizer.pose_adjustment'].shape
            
            # Combine the state dictionaries
            state_dict = combined['optimizers'][opt_name]['state']
            if 0 in state_dict:  # Check if state exists
                exp_avg_chunks = []
                exp_avg_sq_chunks = []
                
                for chunk in chunk_params:
                    if 0 in chunk['optimizers'][opt_name]['state']:
                        chunk_state = chunk['optimizers'][opt_name]['state'][0]
                        if 'exp_avg' in chunk_state and 'exp_avg_sq' in chunk_state:
                            exp_avg_chunks.append(chunk_state['exp_avg'])
                            exp_avg_sq_chunks.append(chunk_state['exp_avg_sq'])
                
                if exp_avg_chunks and exp_avg_sq_chunks:
                    # Concatenate and verify shapes
                    exp_avg = torch.cat(exp_avg_chunks, dim=0)
                    exp_avg_sq = torch.cat(exp_avg_sq_chunks, dim=0)
                    
                    # Verify shapes match
                    if exp_avg.shape[0] != param_shape[0]:
                        logger.warning(f"Shape mismatch for {opt_name}: param={param_shape}, exp_avg={exp_avg.shape}")
                        # Resize optimizer states if needed
                        exp_avg = exp_avg[:param_shape[0]]
                        exp_avg_sq = exp_avg_sq[:param_shape[0]]
                    
                    step = chunk_params[0]['optimizers'][opt_name]['state'][0]['step']
                    
                    combined['optimizers'][opt_name]['state'] = {
                        0: {
                            'exp_avg': exp_avg,
                            'exp_avg_sq': exp_avg_sq,
                            'step': step
                        }
                    }

    # Reset step count
    combined['step'] = 0
    
    # Print total numbers after combining
    total_gaussians = combined['pipeline']['_model.gauss_params.means'].shape[0]
    total_bil_grids = combined['pipeline']['_model.bil_grids.grids'].shape[0] if '_model.bil_grids.grids' in combined['pipeline'] else 0
    logger.info("\nAfter combining:")
    logger.info(f"Total number of gaussians: {total_gaussians:,}")
    logger.info(f"Total number of bilateral grids: {total_bil_grids:,}")
    
    return combined

def main():
    parser = argparse.ArgumentParser(description="Combine multiple chunk checkpoints into one")
    parser.add_argument("--chunks_dir", type=str, required=True, 
                       help="Directory containing chunk outputs (e.g. path/to/output_root/chunks)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save combined checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for processing")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--use_mask", action="store_true",
                       help="Use global visible mask to filter gaussians when combining chunks")
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(args.debug)
    
    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    logger.debug(f"chunks_dir: {chunks_dir}")
    chunks_parent = chunks_dir.parent.name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    combined_dir = output_dir / "nerfstudio_models"
    create_directory(combined_dir)

    # Find all chunk directories
    chunk_dirs = sorted([d for d in chunks_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    if not chunk_dirs:
        raise RuntimeError(f"No chunk directories found in {chunks_dir}")

    # Load parameters from each chunk
    chunk_params = []
    total_gaussians = 0
    total_bil_grids = 0
    
    logger.info("\nGaussian counts per chunk:")
    logger.info("-" * 30)
    
    for chunk_dir in chunk_dirs:
        logger.info(f"\nProcessing {chunk_dir.name}...")
        
        # Find the checkpoint file
        model_dir = chunk_dir / "output" / chunk_dir.parent.parent.name / "splatfacto"
        timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            raise RuntimeError(f"No timestamp directories found in {model_dir}")
        
        latest_timestamp_dir = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
        checkpoint_dir = latest_timestamp_dir / "nerfstudio_models"
        
        checkpoints = list(checkpoint_dir.glob("step-*.ckpt"))
        if not checkpoints:
            raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")
        
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('-')[1]))
        
        # Load and store parameters
        params = load_chunk_checkpoint(latest_checkpoint, args.device, args.use_mask, logger)
        chunk_params.append(params)
        
        # Update totals and print chunk stats
        chunk_gaussians = params['pipeline']['_model.gauss_params.means'].shape[0]
        chunk_bil_grids = params['pipeline']['_model.bil_grids.grids'].shape[0] if '_model.bil_grids.grids' in params['pipeline'] else 0
        total_gaussians += chunk_gaussians
        total_bil_grids += chunk_bil_grids
        
        logger.info(f"{chunk_dir.name}: {chunk_gaussians:,} gaussians")

    logger.info("\nSummary:")
    logger.info("-" * 30)
    logger.info(f"Number of chunks: {len(chunk_params)}")
    logger.info(f"Total gaussians: {total_gaussians:,}")
    logger.info(f"Total bilateral grids: {total_bil_grids:,}")
    
    logger.info(f"\nCombining {len(chunk_params)} chunks...")
    checkpoint = combine_gaussian_models(chunk_params, args.device, logger)

    # Save combined checkpoint
    checkpoint_path = combined_dir / "step-000000000.ckpt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved combined checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main() 