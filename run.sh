#!/bin/bash

# Exit on any error
set -e

# Activate conda environment
echo "Activating bird-sql conda environment..."
conda activate bird-sql

echo "Starting sequential execution of BIRD SQL pipeline..."

echo "Step 1: Generating SQL queries..."
python src/generate.py --input_file data/test_all.jsonl --output_dir output/generations/ --num_gpus 1

echo "Step 2: Processing SQLs and comparing against ground truth..."
python src/process_sqls.py --input_file data/test_all.jsonl --generations_dir output/generations/ --output_dir output/with_results/ --compare_against_gt --sql_timeout 30.0

echo "Step 3: Computing rewards..."
VLLM_USE_V1=0 time python src/reward.py --input_file output/with_results/data_with_results.jsonl --output_dir output/with_rewards --num_gpus 1

echo "Step 4: Running analysis..."
python src/analysis.py --rewards_dir output/with_rewards --gt_sql_file data/test_gold_sqls.txt --output_dir output/analysis --num_cpus 100

echo "Pipeline completed successfully!"