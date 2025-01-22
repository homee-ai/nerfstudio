import os
import shutil
import argparse
from pathlib import Path

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def copy_directory(src, dst, ext: str = "txt"):
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isfile(s) and item.endswith(f'.{ext}'):
                shutil.copy2(s, d)
    else:
        print(f"Warning: Source directory {src} does not exist. Skipping this copy operation.")

def main(args):
    root_path = Path(args.input_path).resolve()
    parent_name = root_path.parent.name
    output_root = root_path.parent / f"{parent_name}_nerfstudio"
    pose_method = args.method
    if args.use_icp:
        pose_method = f"{pose_method}_ICP"

    # Check number of images
    images_dir = output_root / "images"
    num_images = len(list(images_dir.iterdir())) if images_dir.exists() else 0
    const_fps_reset = 100
    sampling_strategy = 'fps'
    if num_images <= const_fps_reset:
        sampling_strategy = 'random'

    cmds = [
        'ns-train splatfacto ',
        f'--data {output_root} ',
        '--max-num-iterations 30000',
    ]

    # Add load-dir argument if resume_path is provided
    if args.resume_path:
        # Check if the path exists and contains checkpoint files
        resume_path = Path(args.resume_path)
        if resume_path.exists():
            # Look for the latest checkpoint file
            checkpoint_files = list(resume_path.glob("step-*.ckpt"))
            if checkpoint_files:
                cmds.append(f'--load-dir {args.resume_path}')
            else:
                print(f"Warning: No checkpoint files found in {args.resume_path}")
        else:
            print(f"Warning: Resume path {args.resume_path} does not exist")

    cmds.extend([
        '--pipeline.model.use-mesh-initialization True',
        '--pipeline.model.combine-mesh-sfm True',
        '--pipeline.model.rasterize-mode antialiased',
        '--pipeline.model.use-scale-regularization False',
        '--pipeline.model.camera-optimizer.mode SO3xR3',
        f'--pipeline.datamanager.train-cameras-sampling-strategy {sampling_strategy}',
        '--pipeline.model.use-bilateral-grid True',
        '--viewer.make-share-url True',
        '--vis viewer+tensorboard',
        'colmap',
        f'--colmap_path "colmap/{pose_method}/0"',
        '--auto_scale_poses False',
        '--center_method none',
        '--orientation_method none',
        '--eval-mode fraction',
        '--train-split-fraction 1.0',
    ])

    full_cmds = ' '.join(cmds)  
    print("run nerfstudio with command: ", full_cmds)  
    os.system(full_cmds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ARKit 3DGS output for nerfstudio training.")
    parser.add_argument("--input_path", help="Path to the root directory of run_arkit_3dgs.sh output")
    parser.add_argument("--method", type=str, default=['arkit'], help="Choose pose optimization methods")
    parser.add_argument("--use_icp", action='store_true', default=False, help="use ICP for mesh and point3D")
    parser.add_argument("--resume_path", type=str, help="Path to the previous training output directory to resume from")
    args = parser.parse_args()
    
    main(args)
