#!/usr/bin/env python3
import os
import re
import fnmatch
import hashlib
import shutil
import logging
from tqdm import tqdm  # For progress bar
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Regex pattern to match model checkpoint filenames.
# Matches filenames like: ind_000000a2_step1000.zip or ind_000000a2.zip
MODEL_REGEX = re.compile(r'^(ind_[0-9a-fA-F]+)(?:_step(\d+))?\.zip$')

def scan_model_files(directory):
    """
    Scans the given directory for model checkpoint files and groups them by base name.
    
    Returns:
        dict: Mapping base names to a list of (filename, step) tuples.
    """
    groups = {}
    all_files = [fname for fname in os.listdir(directory) if fname.endswith(".zip")]
    for fname in tqdm(all_files, desc="Scanning model files"):
        m = MODEL_REGEX.match(fname)
        if m:
            base, step_str = m.groups()
            step = int(step_str) if step_str else 0
            groups.setdefault(base, []).append((fname, step))
    return groups

def cleanup_model_group(directory, base, files_info, bulk=False):
    """
    For a given group of model files (sharing the same base name), 
    identify the checkpoint with the highest step and either prompt the user
    (if not bulk) or automatically delete the older ones.
    """
    files_info.sort(key=lambda x: x[1], reverse=True)
    best_file, best_step = files_info[0]
    print(f"\nFor model '{base}', found the following files:")
    for fname, step in files_info:
        print(f"  {fname} (step: {step})")
    print(f"--> Recommended to keep: {best_file} (step: {best_step})")
    
    if bulk:
        confirm = 'y'
    else:
        confirm = input("Do you want to delete the other files? [y/N]: ").strip().lower()
    
    if confirm == 'y':
        for fname, step in files_info[1:]:
            full_path = os.path.join(directory, fname)
            try:
                os.remove(full_path)
                logger.info(f"Deleted duplicate model file: {full_path}")
            except Exception as e:
                logger.error(f"Error deleting file {full_path}: {e}")
    else:
        logger.info(f"Skipping cleanup for model '{base}'.")

def scan_and_cleanup_models(models_dir, bulk=False):
    """
    Scan the models directory for duplicate checkpoint files and clean them up.
    """
    groups = scan_model_files(models_dir)
    if not groups:
        logger.info("No model checkpoint files found.")
        return

    for base, files_info in tqdm(groups.items(), desc="Processing model groups", total=len(groups)):
        if len(files_info) > 1:
            cleanup_model_group(models_dir, base, files_info, bulk=bulk)
        else:
            logger.info(f"Only one checkpoint found for {base}: {files_info[0][0]} (no cleanup needed).")

def scan_log_files(directory, pattern):
    """
    Scan a directory for files matching a given pattern.
    Returns a list of filenames.
    """
    return [fname for fname in os.listdir(directory) if fnmatch.fnmatch(fname, pattern)]

def cleanup_logs(directory, pattern, keep=1):
    """
    For log files matching the given pattern, keep only the most recent 'keep' files.
    """
    files = scan_log_files(directory, pattern)
    if not files:
        logger.info("No log files matching pattern found.")
        return

    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    if len(files) <= keep:
        logger.info("Number of log files (%d) is within limit (%d).", len(files), keep)
        return

    files_to_remove = files[:-keep]
    for fname in files_to_remove:
        full_path = os.path.join(directory, fname)
        try:
            os.remove(full_path)
            logger.info(f"Deleted log file: {full_path}")
        except Exception as e:
            logger.error(f"Error deleting log file {full_path}: {e}")

def merge_tensorboard_subdirs(config_dir, main_subdir="aggregated", bulk=False):
    """
    Merge all TensorBoard log subdirectories in a given configuration folder
    into a single directory.
    
    Args:
        config_dir (str): The configuration folder (e.g., logs/tensorboard_logs/generation_X/hybrid_XXXX).
        main_subdir (str): The target subdirectory to merge logs into.
        bulk (bool): If True, merge automatically without prompting.
    """
    if not os.path.isdir(config_dir):
        logger.error("Directory %s does not exist.", config_dir)
        return

    subdirs = [d for d in os.listdir(config_dir) if os.path.isdir(os.path.join(config_dir, d))]
    # Ensure the main aggregated directory exists.
    if main_subdir not in subdirs:
        main_dir = os.path.join(config_dir, main_subdir)
        os.makedirs(main_dir, exist_ok=True)
    else:
        main_dir = os.path.join(config_dir, main_subdir)

    # Identify subdirectories to merge (exclude main_subdir)
    merge_dirs = [d for d in subdirs if d != main_subdir]
    if not merge_dirs:
        logger.info("No duplicate TensorBoard subdirectories to merge in %s.", config_dir)
        return

    for sub in tqdm(merge_dirs, desc="Merging TensorBoard subdirectories"):
        sub_path = os.path.join(config_dir, sub)
        logger.info("Merging directory %s into %s", sub_path, main_dir)
        for root, dirs, files in os.walk(sub_path):
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(main_dir, file)
                counter = 1
                base, ext = os.path.splitext(dest_file)
                while os.path.exists(dest_file):
                    dest_file = f"{base}_{counter}{ext}"
                    counter += 1
                shutil.move(src_file, dest_file)
        # Remove the now-empty subdirectory.
        shutil.rmtree(sub_path)
        logger.info("Removed directory %s after merging.", sub_path)

def scan_and_merge_tensorboard_logs(base_tensorboard_dir, bulk=False):
    """
    Scan through the TensorBoard logs directory for configuration folders and merge duplicate subdirectories.
    
    Args:
        base_tensorboard_dir (str): Root directory for TensorBoard logs, e.g., logs/tensorboard_logs/generation_X.
        bulk (bool): If True, automatically merge without prompting.
    """
    if not os.path.isdir(base_tensorboard_dir):
        logger.error("TensorBoard logs directory %s does not exist.", base_tensorboard_dir)
        return

    config_dirs = [os.path.join(base_tensorboard_dir, d) for d in os.listdir(base_tensorboard_dir) if os.path.isdir(os.path.join(base_tensorboard_dir, d))]
    for config_dir in tqdm(config_dirs, desc="Processing configuration directories", total=len(config_dirs)):
        merge_tensorboard_subdirs(config_dir, main_subdir="aggregated", bulk=bulk)

def main():
    parser = argparse.ArgumentParser(description="Scan and cleanup duplicate model checkpoints and TensorBoard logs.")
    parser.add_argument("--bulk", action="store_true", help="Automatically delete/merge duplicates without prompting.")
    args = parser.parse_args()
    
    models_dir = os.path.join(os.getcwd(), "models")
    logs_dir = os.path.join(os.getcwd(), "logs")
    
    print("Scanning for duplicate model checkpoints in:", models_dir)
    scan_and_cleanup_models(models_dir, bulk=args.bulk)
    
    print("\nScanning for duplicate rotated log files...")
    cleanup_logs(logs_dir, "trading_env.log.*", keep=1)
    cleanup_logs(logs_dir, "events.out.tfevents.*", keep=1)
    
    # Merge TensorBoard logs.
    # Assuming TensorBoard logs are stored under: LOG_DIR/tensorboard_logs/generation_X/
    # Adjust the generation number as needed.
    tb_generation_dir = os.path.join(LOG_DIR, "tensorboard_logs", "generation_1")
    print("\nScanning and merging duplicate TensorBoard log subdirectories in:", tb_generation_dir)
    scan_and_merge_tensorboard_logs(tb_generation_dir, bulk=args.bulk)

if __name__ == "__main__":
    main()
