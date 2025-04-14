import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

def split_files_into_groups(source_dir, target_dir_template, num_groups):
    # Get all JSON files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    files.sort()  # Ensure consistent ordering

    # Split files into groups
    groups = [[] for _ in range(num_groups)]
    for idx, file in enumerate(files):
        groups[idx % num_groups].append(file)

    # Create target directories and move files
    for group_id, group_files in enumerate(groups):
        target_dir = target_dir_template.format(group_id)
        os.makedirs(target_dir, exist_ok=True)
        for file in group_files:
            shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))

def main():
    base_dir = "mad_checkpoint"
    num_groups = 10

    # Directories to process
    cur_query_pred_dir = os.path.join(base_dir, "cur_query_pred")
    query_id2windowidx_dir = os.path.join(base_dir, "query_id2windowidx")

    # Ensure the directories exist
    if not os.path.exists(cur_query_pred_dir) or not os.path.exists(query_id2windowidx_dir):
        print("Source directories do not exist.")
        return

    # Get all JSON files in cur_query_pred to determine group partition
    cur_query_pred_files = [f for f in os.listdir(cur_query_pred_dir) if f.endswith('.json')]
    cur_query_pred_files.sort()  # Ensure consistent ordering

    # Split files into groups
    groups = [[] for _ in range(num_groups)]
    for idx, file in enumerate(cur_query_pred_files):
        groups[idx % num_groups].append(file)

    # Move cur_query_pred files
    for group_id, group_files in tqdm(enumerate(groups)):
        target_dir = os.path.join(base_dir, f"cur_query_pred_{group_id}")
        os.makedirs(target_dir, exist_ok=True)
        for file in group_files:
            shutil.move(os.path.join(cur_query_pred_dir, file), os.path.join(target_dir, file))

    # Move query_id2windowidx files using the same group partition
    for group_id, group_files in tqdm(enumerate(groups)):
        target_dir = os.path.join(base_dir, f"query_id2windowidx_{group_id}")
        os.makedirs(target_dir, exist_ok=True)
        for file in group_files:
            source_file = os.path.join(query_id2windowidx_dir, file)
            if os.path.exists(source_file):  # Ensure the file exists before moving
                shutil.move(source_file, os.path.join(target_dir, file))

if __name__ == "__main__":
    main()