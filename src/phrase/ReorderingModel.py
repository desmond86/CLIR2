"""Provides reordering model.

Authors:
Hai Dong Luong (573780) <hai-ld>
Desmond Putra () <dputra>
Andrew Vadnal (326558) <avadnal>
"""


class ReorderingModel:
    """Represent a simple reordering model based on phrase distance.
    
    limit: reordering limit. The distance between phrases should not be larger
    than this number.
    """
    def __init__(self, limit=3):
        self.limit = limit

    def score(self, prev_phrase, next_phrase):
        """Calculate reordering cost based on distance between phrases.

        prev_phrase: previous phrase in translation (not input sentence) order

        next_phrase: next phrase in translation order

        Return reordering cost as a number. If phrases distance is larger than
        this number, return None.

        >>> from collections import defaultdict
        >>> from TranslationOption import TranslationOption
        >>> from datastructures import Hypothesis
        >>> fc_table = defaultdict(lambda: defaultdict(float))
        >>> empty_hypothesis = Hypothesis(
        ...     None, None, 'a b c'.split(), fc_table)
        >>> trans_opt = TranslationOption(1, 2, ['b', 'c'], '2', 0.0)
        >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
        >>> rm = ReorderingModel()
        >>> print rm.score(hypothesis.trans['input'][-1], [(0, 'a')])
        -3
        """
        # the code looks cryptic because phrase's data format is different
        # between Hypothesis (prev_phrase) and utility functions (next_phrase)
        # TODO make an explicit data structure for phrases
        dist = next_phrase[0][0] - prev_phrase[1] - 1
        if dist > self.limit:
            return None
        return -(abs(dist))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
