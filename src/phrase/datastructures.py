# Authors:
# Hai Dong Luong (573780) <hai-ld>
# Desmond Putra () <dputra>
# Andrew Vadnal (326558) <avadnal>

import bisect

# custom modules
from utils import get_consecutive_parts, get_untranslated_words

class Hypothesis:
    def __init__(self, hyp, trans_opt, input_sent=None, fc_table=None):
        """Create a hypothesis expanding on another hypothesis by a
        translation option."""
        #self.prev = hyp # pointer to previous hypothesis
        #self.next = [] # pointers to next hypotheses
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
                self.trans['output'][-1] == other.trans['output'][-1]

    def __lt__(self, other):
        return self.trans['score'] + self.future_cost < \
            other.trans['score'] + other.future_cost

    def __le__(self, other):
        return self.trans['score'] + self.future_cost <= \
                other.trans['score'] + other.future_cost

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)

    def __str__(self):
        return str(self.trans)

    def __repr__(self):
        return str(self.__dict__)


class Stack:
    def __init__(self, size):
        """Create a stack of specified size."""
        self.size = size
        self.hyps = []  # list of hypotheses in ascending order

    def add(self, hyp):
        """Add a hypothesis into the stack."""
        #bisect.insort(self.hyps, hyp)
        #return
        identical_hyps = []
        has_identical = False
        for i, h in enumerate(self.hyps):
            if h.identical(hyp):
                has_identical = True
                if h < hyp:
                    identical_hyps.append(i)

        if identical_hyps:
            for i, j in enumerate(identical_hyps):
                del self.hyps[j - i]

        # This is an example of 'Histogram pruning'
        # If the stack approaches its MAXSIZE, prune it by
        # removing (in this case) one hypothesis - the bottom
        # or 'oldest' element of the stack.
        if len(self.hyps) > self.size:
            del self.hyps[0]

        if identical_hyps or not has_identical:
            bisect.insort(self.hyps, hyp)
            return hyp
        else:
            return None

    def hypotheses(self):
        """Get all hypotheses in the stack."""
        return self.hyps

    def best(self):
        """Return the best hypotheses in the stack."""
        try:
            return self.hyps[-1]
        except IndexError:
            return None