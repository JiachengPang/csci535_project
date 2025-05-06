#!/usr/bin/env bash
modes=(early)

for mode in "${modes[@]}"; do
  echo "Running: python decoder_main.py --mode $mode"
  python decoder_main.py --mode "$mode"
done

echo "Finished training"