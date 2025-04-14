import os
import json

# Input and output file paths
input_file = "data/mad_data_for_cone/data/mad_data/test.jsonl"
output_dir = "data/mad_data_for_cone/data/mad_data/"
os.makedirs(output_dir, exist_ok=True)

# Read all lines from the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Determine the number of lines per fold
num_lines = len(lines)
fold_size = num_lines // 10
remainder = num_lines % 10

# Split lines into 10 folds and write to separate files
start = 0
for i in range(10):
    end = start + fold_size + (1 if i < remainder else 0)
    fold_lines = lines[start:end]
    output_file = os.path.join(output_dir, f"test_{i + 1}.jsonl")
    with open(output_file, 'w') as f:
        f.writelines(fold_lines)
    start = end

print("Data split into 10 folds successfully.")