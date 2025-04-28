#!/usr/bin/env bash
modes=(xnorm late)

for mode in "${modes[@]}"; do
  echo "Running: python decoder_main.py --mode $mode"
  python decoder_main.py --mode "$mode"
done

echo "Finished training"