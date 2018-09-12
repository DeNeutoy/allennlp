from overrides import overrides

from allennlp.training.metrics.metric import Metric
from collections import Counter

@Metric.register("count")
class Count(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just counts unique values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to count the output using our
    ``Metric`` API.
    """
    def __init__(self, name: str) -> None:
        self._counter = Counter()
        self._count = 0.0
        self.name = name

    @overrides
    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The values to count.
        """
        self._counter[list(self.unwrap_to_tensors(value))[0]] += 1
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """

        percentage_dict = {f"{self.name}_{str(k)}": v/self._count if self._count > 0 else 0
                           for k, v in self._counter.items()}

        if reset:
            self.reset()
        return percentage_dict

    @overrides
    def reset(self):
        self._counter = Counter()
        self._count = 0
