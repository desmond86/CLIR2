"""
Provides data structures for decoding process.
# Authors:
# Hai Dong Luong (573780) <hai-ld>
# Desmond Putra () <dputra>
# Andrew Vadnal (326558) <avadnal>
"""

import bisect

# custom modules
from utils import get_consecutive_parts, get_untranslated_words


class Hypothesis:
    """Data structure representing a hypothesis as described in Koehn 06."""
    def __init__(self, hyp, trans_opt, input_sent=None, fc_table=None):
        """Create a hypothesis expanding on another hypothesis by a
        translation option.

        hyp: a hypothesis to be expanded. If None, create an empty hypothesis.

        trans_opt: a TranslationOption. Ignored if hyp is None.

        input_sent: the input sentence. Only needed when create an empty
        hypothesis. New hypothesis's input sentence is set to expanded
        hypothesis's.

        fc_table: future cost table. Only needed when create an empty
        hypothesis. New hypothesis's future cost table is set to expanded
        hypothesis's.
        """
        if hyp is not None:
            #hyp.next.append(self)
            self.input_sent = hyp.input_sent
            self.fc_table = hyp.fc_table
            self.trans = {}
            self.trans['input'] = hyp.trans['input'] + [(
                trans_opt.i_start, trans_opt.i_end, trans_opt.input_phrase)]
            self.trans['output'] = (hyp.trans['output'] + \
                                    [trans_opt.output_phrase])
            self.trans['score'] = hyp.trans['score'] + trans_opt.score
        else:  # create an empty hypothesis
            self.trans = {
                'input': [],
                'output': [],
                'score': 0.0,
            }
            self.input_sent = input_sent
            self.fc_table = fc_table

        self.future_cost = 0.0
        parts = get_consecutive_parts(get_untranslated_words(self))
        if parts[0]:
            for part in parts:
                self.future_cost += self.fc_table[part[0][0]][part[-1][0] + 1]

    def input_len(self):
        """Return the length of input consumed by this hypothesis.

        >>> from collections import defaultdict
        >>> from TranslationOption import TranslationOption
        >>> fc_table = defaultdict(lambda: defaultdict(float))
        >>> empty_hypothesis = Hypothesis(
        ...     None, None, 'a b c'.split(), fc_table)
        >>> trans_opt = TranslationOption(1, 2, ['b', 'c'], '2', 0.0)
        >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
        >>> empty_hypothesis.input_len()
        0
        >>> hypothesis.input_len()
        2
        """
        l = 0
        for i in self.trans['input']:
            l += len(i[2])
        return l

    def identical(self, other):
        """Check whether this hypothesis is identical with another one.

        Two hypotheses are identical when:
        - They consume the same sequence of input. Ordering doesn't matter,
        with the exception of the last word, i.e:
            0, 2, 3, 1 == 2, 3, 0, 1;
        This ensures they have the same set of possible expansions.
        - The last input words' positions are identical, ensuring the same
        reordering cost upon hypothesis expansion.
        - The last output words are identical, ensuring that language model
        scores are identical upon expansion.
        """
        this_i_sequence, other_i_sequence = [], []
        for phrase in self.trans['input']:
            this_i_sequence += range(phrase[0], phrase[1] + 1)
        for phrase in other.trans['input']:
            other_i_sequence += range(phrase[0], phrase[1] + 1)
        this_i_sequence.sort()
        other_i_sequence.sort()

        return this_i_sequence == other_i_sequence and \
                self.trans['input'][-1][1] == other.trans['input'][-1][1] and \
                self.trans['output'][-1].split()[-1] == \
                    other.trans['output'][-1].split()[-1]

    def __lt__(self, other):
        """a.__lt__(b) <==> a < b
        A hypothesis is "less than" another hypothesis when the sum of its
        current score and future cost is less than the other one. Used for
        sorting hypothesis.
        """
        return self.trans['score'] + self.future_cost < \
            other.trans['score'] + other.future_cost

    def __le__(self, other):
        """a.__le__(b) <==> a <= b.

        See __lt__(self, other).
        """
        return self.trans['score'] + self.future_cost <= \
                other.trans['score'] + other.future_cost

    def __gt__(self, other):
        """a.__gt__(b) <==> a > b.

        See __lt__(self, other).
        """
        return not (self <= other)

    def __ge__(self, other):
        """a.__ge__(b) <==> a >= b.

        See __lt__(self, other).
        """
        return not (self < other)

    def __str__(self):
        return str(self.trans)

    def __repr__(self):
        return str(self.__dict__)


class Stack:
    """Data structure representing stacks as described in Koehn 06."""
    def __init__(self, size, pruning_type="Histogram", alpha=None):
        """Create a stack of specified size."""
        self.size = size
        self.hyps = []  # list of hypotheses in ascending order
        self.alpha = alpha
        self.pruning_type = pruning_type

    def add(self, hyp):
        """Add a hypothesis into the stack.

        A hypothesis is added when there is no identical hypothesis or there is
        a worse hypothesis in the stack. Stack gets pruned (histogram) when
        it's over-sized.

        Return True when hypothesis is add, False otherwise.

        >>> from collections import defaultdict
        >>> from TranslationOption import TranslationOption
        >>> fc_table = defaultdict(lambda: defaultdict(float))
        >>> empty_hypothesis = Hypothesis(
        ...     None, None, 'a b c'.split(), fc_table)
        >>> trans_opt = TranslationOption(1, 2, ['b', 'c'], '2', 0.0)
        >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
        >>> stack = Stack(10)
        >>> stack.add(hypothesis)
        True

        >>> trans_opt = TranslationOption(1, 2, ['b', 'c'], '0 3', -1.0)
        >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
        >>> stack.add(hypothesis)
        True

        >>> trans_opt = TranslationOption(1, 2, ['b', 'c'], '1 3', -2.0)
        >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
        >>> # Not added because identical but worse score
        >>> stack.add(hypothesis)
        False

        >>> trans_opt = TranslationOption(1, 2, ['b', 'c'], '2 3', -0.5)
        >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
        >>> # Added because identical but better score
        >>> stack.add(hypothesis)
        True
        """
        idx, identical_hyp = 0, None
        for i, h in enumerate(self.hyps):
            if h.identical(hyp):
                identical_hyp = h
                idx = i
                # there can only be one hypothesis identical to
                # the one being added because there are no existing hypotheses
                # in the stack identical to one another
                break

        if identical_hyp and identical_hyp < hyp:
            del self.hyps[idx]

        if not identical_hyp or (identical_hyp and identical_hyp < hyp):

            if self.pruning_type is "Histogram":
                bisect.insort(self.hyps, hyp)

                # This is an example of 'Histogram pruning'
                # If the stack approaches its MAXSIZE, prune it by
                # removing (in this case) one hypothesis - the top
                # or worst scored element of the stack.
                if len(self.hyps) > self.size:
                    del self.hyps[0]

            elif self.pruning_type is "Threshold":
                try:
                    best_score = self.hyps[-1].trans['score'] + self.hyps[-1].future_cost

                    # If the score of a hypothesis is 'threshold/alpha' times worse than best, prune it
                    # If it is > alpha, we do not add it.
                    if (best_score / (hyp.future_cost + hyp.trans['score'])) < self.alpha:
                        bisect.insort(self.hyps, hyp)
                        
                except IndexError:
                        bisect.insort(self.hyps, hyp)

            return True
        return False

    def hypotheses(self):
        """Get all hypotheses in the stack."""
        return self.hyps

    def best(self):
        """Return the best hypotheses in the stack."""
        try:
            return self.hyps[-1]
        except IndexError:
            return None

if __name__ == '__main__':
    import doctest
    doctest.testmod()
