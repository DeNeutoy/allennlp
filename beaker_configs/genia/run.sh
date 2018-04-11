#!/usr/bin/env bash

source activate allennlp
CONFIG_PATH="beaker_configs/genia"
# Baseline, no elmo.

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                            --memory 10GB --gpu_count 1 --desc "baseline, glove only"

export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
export ELMO="elmo_2x4096_512_2048cnn_2xhighway_options.1"
export NAME="regular elmo"


python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"


export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_Genia_training_sentences.hdf5"
export ELMO="ds_g8ojn0v8uot9"
export NAME="elmo tuned on just genia training data"


python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"

export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_small_5epochs.hdf5"
export ELMO="ds_oavfzynyzb28"
export NAME="elmo tuned on pubmed small data"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"


export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
export ELMO="ds_t3sage6lmxai"
export NAME="elmo trained on 800 m pubmed tokens from scratch"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS"