"""Provides data structure for translation option.

Authors:
Hai Dong Luong (573780) <hai-ld>
Desmond Putra () <dputra>
Andrew Vadnal (326558) <avadnal>
"""


class TranslationOption:
    """Represent translation options."""
    def __init__(self, i_start, i_end, input_phrase, output_phrase, score):
        """Create a translation option.

        i_start: start position of input phrase

        i_end: end position of input phrase

        input_phrase: input phase as a list of words

        output_phrase: output phrase as a string

        score: the score assigned to this translation

        >>> from collections import defaultdict
        >>> from TranslationOption import TranslationOption
        >>> from datastructures import Hypothesis
        >>> fc_table = defaultdict(lambda: defaultdict(float))
        >>> empty_hypothesis = Hypothesis(
        ...     None, None, 'a b c'.split(), fc_table)

        >>> trans_opt = TranslationOption(1, 2, ['b', 'c'], 'B C', 0.0)

        >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
        >>> # 2 because expand option 'b c' => 'B C' on empty hypothesis
        >>> hypothesis.input_len()
        2
        """
        self.i_start = i_start
        self.i_end = i_end
        self.input_phrase = input_phrase
        self.output_phrase = output_phrase
        self.score = score

if __name__ == '__main__':
    import doctest
    doctest.testmod()
