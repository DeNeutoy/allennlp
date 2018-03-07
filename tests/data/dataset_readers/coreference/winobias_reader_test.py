# pylint: disable=no-self-use,invalid-name
from typing import List, Tuple

import pytest

from allennlp.data.dataset_readers import WinobiasReader
from allennlp.common.util import ensure_list

class TestWinobiasReader:
    span_width = 5

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = WinobiasReader(max_span_width=self.span_width, lazy=lazy)
        instances = ensure_list(conll_reader.read("tests/fixtures/coref/winobias.sample"))

        assert len(instances) == 2

        fields = instances[0].fields
        text = [x.text for x in fields["text"].tokens]

        assert text == []

        spans = fields["spans"].field_list
        span_starts, span_ends = zip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields["span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids: List[Tuple[List[str], int]] = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]

        fields = instances[1].fields

    def check_candidate_mentions_are_well_defined(self, span_starts, span_ends, text):
        candidate_mentions = []
        for start, end in zip(span_starts, span_ends):
            # Spans are inclusive.
            text_span = text[start: end + 1]
            candidate_mentions.append(text_span)

        # Check we aren't considering zero length spans and all
        # candidate spans are less than what we specified
        assert all([self.span_width >= len(x) > 0 for x in candidate_mentions])  # pylint: disable=len-as-condition
        return candidate_mentions
