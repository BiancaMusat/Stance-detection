#!/usr/bin/env bash

echo "Generating dynamics from previously computed logits. You can provide the name of the logits file, the number of epochs and the number of batches"
python training_dynamics.py -f $1 -e $2