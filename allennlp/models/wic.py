from typing import Dict, List, Optional, Any, Union

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy


@Model.register("wic")
class Wic(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model : ``Union[str, BertModel]``, required.
        A string describing the BERT model to load or an already constructed BertModel.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self._accuracy = BooleanAccuracy()
        self._classification_layer = Linear(self.bert_model.config.hidden_size, 1)

        self._dropout = Dropout(p=dropout)
        self._loss = torch.nn.BCEWithLogitsLoss()
        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: Dict[str, torch.Tensor],
        new_token_type_ids: torch.Tensor,
        metadata: List[Any],
        label: torch.LongTensor = None,
    ):

        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence, the verb to compute the
            frame for, and start offsets for converting wordpieces back to a sequence of words,
            under 'words', 'verb' and 'offsets' keys, respectively.

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
        mask = get_text_field_mask(tokens)
        bert_embeddings, pooled = self.bert_model(
            input_ids=tokens["tokens"],
            token_type_ids=new_token_type_ids,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )

        pooled = self._dropout(pooled)

        # apply classification layer
        logits = self._classification_layer(pooled)

        probs = torch.sigmoid(logits)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits.view(-1), label.float().view(-1))
            output_dict["loss"] = loss
            self._accuracy((probs > 0.5).view(-1).long(), label.view(-1))

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        return output_dict

    def get_metrics(self, reset: bool = False):

        return {"accuracy": self._accuracy.get_metric(reset)}
