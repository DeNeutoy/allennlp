import torch
from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.data.fields import IndexField, TextField
from allennlp.models import SemanticRoleLabeler
from allennlp.data.token_indexers import SingleIdTokenIndexer

model_config = {
    "type":"srl",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "/net/efs/aristo/dlfa/glove/glove.6B.100d.txt.gz",
        "trainable": True
      }
    },
    "initializer": {
            "default": { "type": "normal", "std": 0.01},
            "exclude": ["token_embedder_tokens"]
    },
    "stacked_encoder": {
      "type": "lstm",
      "input_size": 101,
      "hidden_size": 128,
      "num_layers": 4
    }
}

model_weights_path = "/net/efs/aristo/allennlp/srl/srl-model3/model_state_epoch_49.th"
vocab_path = "/net/efs/aristo/allennlp/srl/srl-model3/vocabulary/"

vocab = Vocabulary.from_files(vocab_path)
model = SemanticRoleLabeler.from_params(vocab, Params(model_config))
model_weights = torch.load(model_weights_path)
model.load_state_dict(model_weights)

demo_text = TextField(["Michael", "was", "excited", "to", "see", "some", "SRL", "output", "!"],
                      {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)})
demo_verb = IndexField(2, demo_text)

output = model.tag(demo_text, demo_verb)
print("Sentence: ", " ".join(demo_text.tokens()))
print("Predictions: ", output["tags"])





