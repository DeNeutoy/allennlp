from typing import Dict, Tuple, List
import logging
import os

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
import glob

logger = logging.getLogger(__name__)


@DatasetReader.register("snli")
class WicReader(DatasetReader):

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        bert_model_name: str = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False



    @overrides
    def _read(self, file_path: str):

        data_path = glob.glob(os.path.join(file_path, "data.txt"))
        if not data_path:
            raise ValueError("Could not find any data.")

        label_path = glob.glob(os.path.join(file_path, "gold.txt"))
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as snli_file:
            logger.info("Reading SNLI instances from jsonl dataset at: %s", file_path)


    @overrides
    def text_to_instance(
        self,  # type: ignore
        sentence1: List[str],
        sentence2: List[str],
        index1: int,
        index2: int,
        label: str = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        index2 += len(sentence1) + 1

        combined_sentences = sentence1 + ["[SEP]"] + sentence2
        wordpieces1, offsets1, start_offsets1 = self._wordpiece_tokenize_input(
            combined_sentences
        )
        fields["sentences"] = TextField(
            [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces1],
            token_indexers=self._token_indexers,
        )

        token_type_ids = [0 for _ in wordpieces1]
        token_type_ids[index1] =  1
        token_type_ids[index2] =  1
        fields["new_offsets"] = SequenceLabelField(token_type_ids, fields["sentences"])
        if label:
            fields["label"] = LabelField(label)

        metadata = {
            "sent1": sentence1,
            "sent2": sentence2,
            "index1": index1,
            "index2": index2,
            "label": label
        }
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    def _wordpiece_tokenize_input(
        self, tokens: List[str]
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.

        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.

        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).

        For example:

        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).

        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets
