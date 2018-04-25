#!/usr/bin/env bash

source activate allennlp

SEED=$1
CONFIG_PATH="beaker_configs/genia"
PUBMED_GLOVE="ds_33fmprd6uv1g"
# Baseline, no elmo.
python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline.json --source "glove.1:/glove" --source "genia-ptb:/genia-ptb" \
                                                            --memory 10GB --gpu_count 1 --desc "baseline, glove only" --name "baseline-glove-only-$SEED" --env "SEED=$SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_pubmed.json --source "$PUBMED_GLOVE:/glove" --source "genia-ptb:/genia-ptb" \
                                                            --memory 10GB --gpu_count 1 --desc "baseline, glove only" --name "baseline-bio-glove-only-$SEED" --env "SEED=$SEED"

export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
export ELMO="elmo_2x4096_512_2048cnn_2xhighway_options.1"
export NAME="regular elmo $SEED"


python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME" --name "$NAME"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "$PUBMED_GLOVE:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with bio glove" --name "$NAME with bio glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"


export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_Genia_training_sentences.hdf5"
export ELMO="ds_g8ojn0v8uot9"
export NAME="elmo tuned on just genia training data $SEED"


python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME" --name "$NAME" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "$PUBMED_GLOVE:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with bio glove" --name "$NAME with bio glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_small_5epochs.hdf5"
export ELMO="ds_oavfzynyzb28"
export NAME="elmo tuned on pubmed small data $SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME" --name "$NAME" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "$PUBMED_GLOVE:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with bio glove" --name "$NAME with bio glove"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_PubMed_sentences_large_5epochs.hdf5"
export ELMO="ds_lve4gevjgfji"
export NAME="elmo tuned on pubmed large data $SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME" --name "$NAME"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "$PUBMED_GLOVE:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with bio glove" --name "$NAME with bio glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

export ELMO_OPTIONS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
export ELMO_WEIGHTS="/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
export ELMO="ds_t3sage6lmxai"
export NAME="elmo trained on 800 m pubmed tokens from scratch $SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo.json --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME" --name "$NAME"\
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"

python scripts/ai2-internal/run_with_beaker.py $CONFIG_PATH/baseline_elmo_glove.json --source "$PUBMED_GLOVE:/glove" --source "genia-ptb:/genia-ptb" \
                                                             --source "$ELMO:/elmo" --memory 10GB --gpu_count 1 --desc "$NAME with bio glove" --name "$NAME with bio glove" \
                                                             --env "ELMO_OPTIONS=$ELMO_OPTIONS" --env "ELMO_WEIGHTS=$ELMO_WEIGHTS" --env "SEED=$SEED"
