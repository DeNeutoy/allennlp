from typing import List, Tuple, Dict
import argparse
from collections import defaultdict

from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes

train_path = "/Users/markn/allen_ai/data/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train"
dev_path = "/Users/markn/allen_ai/data/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/dev"
test_path = "/Users/markn/allen_ai/data/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/test"

LabelSpan = Tuple[str, Tuple[int, int]]


def bio_tags_to_spans(tag_sequence: List[str],
                      classes_to_ignore: List[str] = None) -> List[LabelSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans.

    Parameters
    ----------
    tag_sequence : List[str], required.
        The string class labels for a sequence.
    classes_to_ignore : List[str], optional (default = None).
        A list of string class labels which should be ignored
        when extracting spans.

    Returns
    -------
    spans : Set[Tuple[Tuple[int, int], str]]
        The typed, extracted spans from the sequence, in the format ((span_start, span_end), label).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "U":
            # The U tag is used to indicate a span of length 1,
            # so if there's an active tag we end it, and then
            # we add a "length 0" tag.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            spans.add((conll_tag, (index, index)))
            active_conll_tag = None
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def inside_outside_overlaps(span: Tuple[int, int],
                            span_list: List[Tuple[int, int]]) -> str:
    is_inside = False
    overlaps = None
    for target_span in span_list:

        # Start inbetween target.
        if target_span[0] <= span[0] <= target_span[1]:
            if span[1] <= target_span[1]:
                is_inside = True
            else:
                overlaps = True

        # begining before, end inside
        if span[0] < target_span[0] and span[1] in range(target_span[0], target_span[1] + 1):
            overlaps = True
        # begining inside, end after
        if span[0] in range(target_span[0], target_span[1] + 1) and span[1] > target_span[1]:
            overlaps = True

    if is_inside:
        return "inside"
    elif overlaps:
        return "overlaps"
    else:
        return "outside"


def accumulate_statistics(stats: Dict[str, int],
                          coref_spans: List[Tuple[int, Tuple[int, int]]],
                          srl_spans: List[LabelSpan],
                          ner_spans: List[LabelSpan]):

    untyped_srl_spans = [x[1] for x in srl_spans]
    for span in coref_spans:
        stats["coref_spans"] += 1

        if not untyped_srl_spans:
            stats["coref_no_srl_spans"] += 1

        elif span[1] in untyped_srl_spans:
            stats["coref_exact_match"] += 1

        else:
            containment = inside_outside_overlaps(span[1], untyped_srl_spans)
            stats["coref_" + containment] += 1

    for span in ner_spans:
        stats["ner_spans"] += 1

        if not untyped_srl_spans:
            stats["ner_no_srl_spans"] += 1

        elif span[1] in untyped_srl_spans:
            stats["ner_exact_match"] += 1
        else:
            containment = inside_outside_overlaps(span[1], untyped_srl_spans)
            stats["ner_" + containment] += 1

def main(path):
    ontonotes_reader = Ontonotes()

    stats = defaultdict(int)

    for sentence in ontonotes_reader.dataset_iterator(path):

        coref_spans = list(sentence.coref_spans)
        ner_spans = bio_tags_to_spans(sentence.named_entities)
        srl_spans = []
        for srl_frame in sentence.srl_frames.values():
            srl_spans.extend(bio_tags_to_spans(srl_frame))

        accumulate_statistics(stats, coref_spans, srl_spans, ner_spans)

    print(f"NER Exact Span Match, All SRL Roles: {stats['ner_exact_match']/stats['ner_spans'] * 100} %")
    print(f"NER No SRL information : {stats['ner_no_srl_spans']/stats['ner_spans'] * 100} %")
    print("------------------------")
    for containment in ["inside", "outside", "overlaps"]:
        print(f"NER {containment} a Span, All SRL Roles: {stats['ner_' + containment]/stats['ner_spans'] * 100} %")

    print(f"NER Total inside, exact match or no info : "
          f"{(stats['ner_exact_match'] + stats['ner_inside'] + stats['ner_no_srl_spans'])/stats['ner_spans'] * 100} %")

    print("------------------------")
    print(f"Coref Exact Span Match, All SRL Roles: {stats['coref_exact_match']/stats['coref_spans'] * 100} %")
    print(f"Coref No SRL information : {stats['coref_no_srl_spans']/stats['coref_spans'] * 100} %")
    print("------------------------")
    for containment in ["inside", "outside", "overlaps"]:
        print(f"Coref {containment} a Span, All SRL Roles: {stats['coref_' + containment]/stats['coref_spans'] * 100} %")
    print(f"Coref Total inside, exact match or no info : "
          f"{(stats['coref_exact_match'] + stats['coref_inside'] + stats['coref_no_srl_spans'])/stats['coref_spans'] * 100} %")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ontonotes span analysis")
    parser.add_argument('--path', type=str, help='The dataset directory.')

    args = parser.parse_args()
    main(args.path)

