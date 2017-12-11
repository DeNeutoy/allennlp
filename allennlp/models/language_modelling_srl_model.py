from typing import Dict, Optional

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn.initializers import InitializerApplicator
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.modules.scalar_mix import ScalarMix


@Model.register("language-modelling-srl")
class LanguageModelingSemanticRoleLabeler(Model):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it extends `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_  by
    adding pretrained bidirectional language modelling embeddings into the inputs and outputs
    of the stacked LSTMs.

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    The bidirectional language models are incorporated by using a scalar linear combination of
    the three bidirectional embeddings at the input and output of the stacked LSTM. The scalar
    parameters for the input and output embeddings are learned separately, but the embeddings
    themselves are fixed and not trained.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    stacked_encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    language_model_feature_dim: int, required.
        The dimensionality of the bidirectional language model embeddings.
    initializer : ``InitializerApplicator``
        We will use this to initialize the parameters in the model, calling ``initializer(self)``.
    embedding_dropout: float, optional (default = 0.0)
        The dropout probability for the learnt word embedding.
    language_model_embedding_dropout: float, optional (default = 0.0)
        The dropout probability for the bidirectional language model embedding features.
    use_input_language_model: bool, (default = True)
        Whether or not to use the language model features in the input to the stacked LSTMs.

    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 stacked_encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 language_model_feature_dim: int,
                 initializer: InitializerApplicator,
                 embedding_dropout: float = 0.0,
                 language_model_embedding_dropout: float = 0.5,
                 use_input_language_model: bool = True) -> None:
        super(LanguageModelingSemanticRoleLabeler, self).__init__(vocab)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels", ignore_classes=["V"])

        self.stacked_encoder = stacked_encoder
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)

        output_projection_size = self.stacked_encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_projection_size,
                                                           self.num_classes))

        self.input_language_model_mixture = ScalarMix(3, do_layer_norm=True)

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.language_model_dropout = Dropout(p=language_model_embedding_dropout)

        self.use_input_language_model = use_input_language_model
        initializer(self)

        encoder_input_dim = text_field_embedder.get_output_dim() + binary_feature_dim
        if use_input_language_model:
            encoder_input_dim += language_model_feature_dim
        if encoder_input_dim != stacked_encoder.get_input_dim():
            raise ConfigurationError("The LM SRL Model uses a binary verb indicator feature and pretrained"
                                     "bidirectional language model embeddings, meaning that the input dimension "
                                     "of the stacked_encoder must be equal to the output dimension of the "
                                     "text_field_embedder + language_model_feature_dim + binary_feature_dim")

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                language_model_embeddings: torch.FloatTensor,
                verb_indicator: torch.LongTensor,
                tags: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        language_model_embeddings : torch.FloatTensor, required.
            The embedding of the sentence generated from the output of a 3 layer bidirectional language
            model at every layer. Has shape: (batch_size, num_layers(3), sentence_length, 1024).
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        _, sequence_length = mask.size()

        per_layer_embeddings = [x.squeeze(1) for x in language_model_embeddings.split(1, 1)]
        if self.use_input_language_model:
            input_language_model_embeddings = self.input_language_model_mixture(per_layer_embeddings, mask) * mask.unsqueeze(-1).float()
            input_language_model_embeddings = self.language_model_dropout(input_language_model_embeddings)

            # Concatenate the verb feature onto the embedded text. This now has shape
            # (batch_size, sequence_length, embedding_dim + binary_feature_dim + language_model_feature_dim).
            encoder_input_embedding = torch.cat([embedded_text_input,
                                                 embedded_verb_indicator,
                                                 input_language_model_embeddings], -1)
        else:
            # (batch_size, sequence_length, embedding_dim + binary_feature_dim).
            encoder_input_embedding = torch.cat([embedded_text_input,
                                                 embedded_verb_indicator], -1)

        batch_size, sequence_length, _ = encoder_input_embedding.size()
        encoded_text = self.stacked_encoder(encoder_input_embedding, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs).view([batch_size, sequence_length, self.num_classes])
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            self.span_metric(class_probabilities, tags, mask)
            output_dict["loss"] = loss

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].data.cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, _ = viterbi_decode(predictions[:length], transition_matrix)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        if len(all_tags) == 1:
            all_tags = all_tags[0]  # type: ignore
        output_dict['tags'] = all_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.span_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # During training, we only really care about the overall
            # metrics, so we filter for them here.
        return {x: y for x, y in metric_dict.items() if "overall" in x}

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'LanguageModelingSemanticRoleLabeler':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        stacked_encoder = Seq2SeqEncoder.from_params(params.pop("stacked_encoder"))
        binary_feature_dim = params.pop("binary_feature_dim")
        language_modeling_feature_dim = params.pop("language_modeling_feature_dim")
        language_model_embedding_dropout = params.pop("language_model_embedding_dropout", 0.5)
        embedding_dropout = params.pop("embedding_dropout", 0.0)
        use_input_language_model = params.pop("use_input_language_model", True)
        initializer = InitializerApplicator.from_params(params.pop("initializer", []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   stacked_encoder=stacked_encoder,
                   binary_feature_dim=binary_feature_dim,
                   language_model_feature_dim=language_modeling_feature_dim,
                   initializer=initializer,
                   embedding_dropout=embedding_dropout,
                   language_model_embedding_dropout=language_model_embedding_dropout,
                   use_input_language_model=use_input_language_model)

