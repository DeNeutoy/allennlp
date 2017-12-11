import codecs
import os
import logging
from typing import Dict, List, Optional

from overrides import overrides
import tqdm
import h5py

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields.array_field import ArrayField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def load_language_model_embeddings(embedding_file: str):
    """
    Load the language model embeddings from the file. The file is keyed by sentence_id.
    Each sentence contains one h5py dataset of shape (3, n_tokens, 1024),
    with the embeddings for each of the three layers in the language model.

    Parameters
    ----------
    embedding_file : str, required.
        The path to the hdf5 file containing a sentence_id -> numpy.float32 array.

    Returns
    -------
    A dictionary mapping sentence_ids -> tensors of shape (3, len(sentence), 1024).

    """
    language_model_embeddings = {}
    with h5py.File(embedding_file, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            sentence_id = key.replace(':', '/')
            language_model_embeddings[sentence_id] = hdf5_file[key][...].astype("float32")

    return language_model_embeddings


@DatasetReader.register("language-modelling-srl")
class LanguageModellingSrlReader(DatasetReader):
    """
    Returns the same data as the SrlReader in Allennlp, but also expects
    that each data directory will have a file called "lm_embedding.hdf5" which
    contains the serialized language model embeddings of each sentence in the SRL data.

    Parameters
    ----------
    return_embedded_arrays : bool, (default = True)
        Whether the DatasetReader should return the actual embeddings as ``ArrayFields`` or the
        metadata key for looking up the embedding.
    embedding_type: str, (default = "lm_embeddings")
        Which pretrained embedding file to use.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.
    """
    def __init__(self,
                 return_embedded_arrays: bool = True,
                 embedding_type: str = "lm_embeddings.hdf5",
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:

        self._return_embedded_arrays = return_embedded_arrays
        self._embedding_type = embedding_type
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _process_sentence(self,
                          sentence_id: str,
                          sentence_tokens: List[str],
                          verbal_predicates: List[int],
                          predicate_argument_labels: List[List[str]]) -> List[Instance]:
        """
        Parameters
        ----------
        sentence_id : ``str``, required.
            The sentence id used to identify the sentence in the serialized
            pretrained language modeling vector dump.
        sentence_tokens : ``List[str]``, required.
            The tokenised sentence.
        verbal_predicates : ``List[int]``, required.
            The indexes of the verbal predicates in the
            sentence which have an associated annotation.
        predicate_argument_labels : ``List[List[str]]``, required.
            A list of predicate argument labels, one for each verbal_predicate. The
            internal lists are of length: len(sentence).
        Returns
        -------
        A list of Instances.
        """
        tokens = [Token(t) for t in sentence_tokens]
        if not verbal_predicates:
            # Sentence contains no predicates.
            tags = ["O" for _ in sentence_tokens]
            verb_label = [0 for _ in sentence_tokens]
            return [self.text_to_instance(tokens, verb_label, sentence_id, tags)]
        else:
            instances = []
            for verb_index, annotation in zip(verbal_predicates, predicate_argument_labels):
                tags = annotation
                verb_label = [0 for _ in sentence_tokens]
                verb_label[verb_index] = 1
                instances.append(self.text_to_instance(tokens, verb_label, sentence_id, tags))
            return instances

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instances = []

        sentence: List[str] = []
        sentence_id: str = None
        verbal_predicates: List[int] = []
        predicate_argument_labels: List[List[str]] = []
        current_span_label: List[Optional[str]] = []

        if self._return_embedded_arrays:
            logger.info("Loading pretrained language model embeddings. This might take a while.")
            self.language_model_embedding_dict = load_language_model_embeddings(
                os.path.join(file_path, self._embedding_type))

        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        for root, _, files in tqdm.tqdm(list(os.walk(file_path))):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every file will be duplicated
                # - one file called filename.gold_skel and one generated from the preprocessing
                # called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue

                # Starting a new file, so reset the sentence_id_counter.
                sentence_id_count = 0
                with codecs.open(os.path.join(root, data_file), 'r', encoding='utf8') as open_file:
                    for line in open_file:
                        line = line.strip()
                        if line == '' or line.startswith("#"):

                            # Conll format data begins and ends with lines containing a hash,
                            # which may or may not occur after an empty line. To deal with this
                            # we check if the sentence is empty or not and if it is, we just skip
                            # adding instances, because there aren't any to add.
                            if not sentence:
                                continue
                            instances.extend(self._process_sentence(sentence_id + "/" + str(sentence_id_count),
                                                                    sentence,
                                                                    verbal_predicates,
                                                                    predicate_argument_labels))
                            # Reset everything for the next sentence.
                            sentence = []
                            sentence_id: str = None
                            verbal_predicates = []
                            predicate_argument_labels = []
                            current_span_label = []
                            sentence_id_count += 1
                            continue

                        conll_components = line.split()
                        word = conll_components[3]
                        # This is the "bc/cnn/00/cnn_0001" bit in every file.
                        # It uniquely identifies the _file_ in the dataset,
                        # so we use it plus a sentence index to uniquely identify
                        # sentences in the dataset.
                        sentence_id = conll_components[0]

                        sentence.append(word)
                        word_index = len(sentence) - 1
                        if word_index == 0:
                            # We're starting a new sentence. Here we set up a list of lists
                            # for the BIO labels for the annotation for each verb and create
                            # a temporary 'current_span_label' list for each annotation which
                            # we will use to keep track of whether we are beginning, inside of,
                            # or outside a particular span.
                            predicate_argument_labels = [[] for _ in conll_components[11:-1]]
                            current_span_label = [None for _ in conll_components[11:-1]]

                        num_annotations = len(predicate_argument_labels)
                        is_verbal_predicate = False
                        # Iterate over all verb annotations for the current sentence.
                        for annotation_index in range(num_annotations):
                            annotation = conll_components[11 + annotation_index]
                            label = annotation.strip("()*")

                            if "(" in annotation:
                                # Entering into a span for a particular semantic role label.
                                # We append the label and set the current span for this annotation.
                                bio_label = "B-" + label
                                predicate_argument_labels[annotation_index].append(bio_label)
                                current_span_label[annotation_index] = label

                            elif current_span_label[annotation_index] is not None:
                                # If there's no '(' token, but the current_span_label is not None,
                                # then we are inside a span.
                                bio_label = "I-" + current_span_label[annotation_index]
                                predicate_argument_labels[annotation_index].append(bio_label)
                            else:
                                # We're outside a span.
                                predicate_argument_labels[annotation_index].append("O")

                            # Exiting a span, so we reset the current span label for this annotation.
                            if ")" in annotation:
                                current_span_label[annotation_index] = None
                            # If any annotation contains this word as a verb predicate,
                            # we need to record its index. This also has the side effect
                            # of ordering the verbal predicates by their location in the
                            # sentence, automatically aligning them with the annotations.
                            if "(V" in annotation:
                                is_verbal_predicate = True

                        if is_verbal_predicate:
                            verbal_predicates.append(word_index)

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         sentence_id: str,
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label and a sentence_id.  The verb
        label should be a one-hot binary vector, the same length as the tokens, indicating the
        position of the verb to find arguments for. The sentence_id uniquely references the sentence
        in the dataset which this example was generated from and is used to lookup language model
        embeddings for the sentence, used in the model.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        if self._return_embedded_arrays:
            fields["language_model_embeddings"] = ArrayField(self.language_model_embedding_dict[sentence_id])
        else:
            fields["sentence_id"] = MetadataField(sentence_id)
        if tags:
            fields['tags'] = SequenceLabelField(tags, text_field)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'SrlReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        return_embedded_arrays = params.pop("return_embedded_arrays", True)
        embedding_type = params.pop("embedding_type", "lm_embeddings.hdf5")
        params.assert_empty(cls.__name__)
        return LanguageModellingSrlReader(return_embedded_arrays,
                                          embedding_type=embedding_type,
                                          token_indexers=token_indexers)
