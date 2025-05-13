#!/bin/bash

datasets=("nq_easy" "nq_hard" "popqa_easy" "popqa_hard" "hotpotqa_easy" "hotpotqa_hard" "2wikimultihopqa_easy" "2wikimultihopqa_hard")

for dataset in "${datasets[@]}"; do
  echo "Processing dataset: $dataset"
  python scripts/data_process/nq_search.py --dataset_name "$dataset"
  echo "Finished processing dataset: $dataset"
  echo "----------------------------------------"
done

echo "Successfully processed all datasets."