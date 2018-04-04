"""
The ``dry-run`` command creates a vocabulary, informs you of
dataset statistics and other training utilities without actually training
a model.

.. code-block:: bash
"""
from typing import Dict, List
from collections import defaultdict
import argparse
import logging
import os

import numpy

from allennlp.commands.train import datasets_from_params
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data.vocabulary import DEFAULT_NON_PADDED_NAMESPACES
from allennlp.data import Vocabulary, Instance
from allennlp.data.fields import SequenceField, ListField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DryRun(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Create a vocabulary, compute dataset statistics and other training utilities.'''
        subparser = parser.add_parser(
                name, description=description, help='Create a vocabulary')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model and its inputs')
        subparser.add_argument('-s', '--serialization-dir',
                               required=True,
                               type=str,
                               help='directory in which to save the output of the dry run.')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.set_defaults(func=dry_run_from_args)

        return subparser


def dry_run_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to params.
    """
    parameter_path = args.param_path
    serialization_dir = args.serialization_dir
    overrides = args.overrides

    params = Params.from_file(parameter_path, overrides)

    dry_run_from_params(params, serialization_dir)

def dry_run_from_params(params: Params, serialization_dir: str) -> None:
    prepare_environment(params)

    vocab_params = params.pop("vocabulary", {})
    vocab_dir = vocab_params.get('directory_path')

    if vocab_dir is not None:
        logger.info("Found a directory_path parameter in your config. "
                    "Also saving the vocab we create to that location.")
        os.makedirs(vocab_dir, exist_ok=True)

    os.makedirs(serialization_dir, exist_ok=True)

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_for_vocab_creation))

    instances = [instance for key, dataset in all_datasets.items()
                 for instance in dataset
                 if key in datasets_for_vocab_creation]

    vocabulary = verbosely_create_vocabulary(vocab_params, instances)

    logger.info(f"writing the vocabulary to {serialization_dir}.")
    vocabulary.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    if vocab_dir is not None and os.path.exists(vocab_dir) and os.listdir(vocab_dir) is not None:
        logger.info(f"You passed a directory_path in your config which already exists and is non-empty. "
                    f"Refusing to overwrite - we saved it to {serialization_dir} instead.")
    elif vocab_dir is not None:
        logger.info(f"You passed a directory_path in your config which was empty. Also "
                    f"writing the vocabulary to {vocab_dir}.")
        vocabulary.save_to_files(vocab_dir)


def verbosely_create_vocabulary(params: Params, instances: List[Instance]) -> Vocabulary:

    min_count = params.pop("min_count", None)
    max_vocab_size = params.pop_int("max_vocab_size", None)
    non_padded_namespaces = params.pop("non_padded_namespaces", DEFAULT_NON_PADDED_NAMESPACES)
    pretrained_files = params.pop("pretrained_files", {})
    only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
    params.assert_empty("Vocabulary - from dataset")

    logger.info("Fitting token dictionary from dataset.")
    namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sequence_field_lengths: Dict[str, List] = defaultdict(list)

    for instance in instances:
        instance.count_vocab_items(namespace_token_counts)
    vocabulary = Vocabulary(counter=namespace_token_counts,
                            min_count=min_count,
                            max_vocab_size=max_vocab_size,
                            non_padded_namespaces=non_padded_namespaces,
                            pretrained_files=pretrained_files,
                            only_include_pretrained_words=only_include_pretrained_words)

    for instance in instances:
        instance.index_fields(vocabulary)
        for key, value in instance.get_padding_lengths().items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sequence_field_lengths[f"{key}.{sub_key}"].append(sub_value)
            else:
                sequence_field_lengths[key].append(value)

    print("\n\n----Dataset Statistics----\n")
    for name, lengths in sequence_field_lengths.items():
        print(f"Statistics for {name}:")
        print(f"\tLengths: Mean: {numpy.mean(lengths)}, Standard Dev: {numpy.std(lengths)}, "
              f"Max: {numpy.max(lengths)}, Min: {numpy.min(lengths)}")

    print("\n10 Random instances: ")
    for i in list(numpy.random.randint(len(instances), size=10)):
        print(f"Instance {i}:")
        print(f"\t{instances[i]}")

    print("\n\n----Vocabulary Statistics----\n")
    print(vocabulary)

    for namespace in namespace_token_counts:
        tokens_with_counts = list(namespace_token_counts[namespace].items())
        tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 most frequent tokens in namespace '{namespace}':")
        for token, freq in tokens_with_counts[:10]:
            print(f"\tToken: {token}\t\tFrequency: {freq}")
        # Now sort by token length, not frequency
        tokens_with_counts.sort(key=lambda x: len(x[0]), reverse=True)

        print(f"\nTop 10 longest tokens in namespace '{namespace}':")
        for token, freq in tokens_with_counts[:10]:
            print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")

        print(f"\nTop 10 shortest tokens in namespace '{namespace}':")
        for token, freq in reversed(tokens_with_counts[-10:]):
            print(f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")

    return vocabulary
