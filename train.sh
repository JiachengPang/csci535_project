#!/usr/bin/env bash
modes=(early late xnorm)

for mode in "${modes[@]}"; do

  echo "Running: python main.py --model $mode"
  python main.py --model "$mode"

  echo "Running: python decoder_main.py --model $mode"
  python decoder_main.py --model "$mode"

done

echo "Running: python train_test_at.py"
python train_test_at.py

echo "Running: python generate.py"
python generate.py --model mbt

echo "Finished training"