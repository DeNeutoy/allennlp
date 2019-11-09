local bert_model = "allennlp/tests/fixtures/bert/vocab.txt";

{
    "dataset_reader": {
        "lazy": false,
        "type": "wic",
        "bert_model_name": "bert-base-uncased",
    },
    "train_data_path": "allennlp/tests/fixtures/data/wic/",
    "validation_data_path": "allennlp/tests/fixtures/data/wic",
    "model": {
        "type": "wic",
        "bert_model": bert_model,
        "dropout": 0.0
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "padding_noise": 0.0,
        "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
