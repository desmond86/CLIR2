# -*- coding: utf-8 -*-
# Authors:
# Hai Dong Luong (573780) <hai-ld>
# Desmond Putra (555802) <dputra>
# Andrew Vadnal (326558) <avadnal>

"""
Assignment 2. A Phrase-based translation model and decoder.

>>> all_file = "source_files/all.lm"
>>> e_file = "source_files/all.test.lowercased.raw.en"
>>> f_file = "source_files/all.test.lowercased.raw.fr"
>>> a_file = "source_files/aligned.grow-diag-final-and"
>>> max_stack_size = 10
>>> decoder = Decoder(all_file, e_file, f_file, a_file, max_stack_size)
>>> decoder.process_models()
>>> alpha = 1.0/2
>>> prune_type = "Threshold"
>>> sentences = 5
>>> decoder.decoder_test(f_file, sentences, prune_type, alpha)
Translating phrase 1 of 5
le rapport von wogau propose de renvoyer le contrôle de cette légitimité à l ' échelon national .
the von wogau report proposes referring supervision of
-596.808846759
Translating phrase 2 of 5
Translating phrase 3 of 5
nous demandons que la plus grande transparence soit de mise et qu ' un plus grand pouvoir d ' investigation soit donné à la commission pour vérifier a posteriori la légitimité de ces exceptions .
we demand that the greatest possible transparency should be in place and that greater powers of investigation be granted to the commission in order to check the legitimacy of such exceptions after the fact
-2391.51000566
Translating phrase 4 of 5
monsieur le président , nous débattons une fois de plus de la politique européenne de concurrence .
mr president , once again we are debating the european union ' s competition
-313.90218715
Translating phrase 5 of 5
mais à vrai dire , dans quelles conditions a lieu ce débat et à quelles conclusions devrions-nous aboutir ?
but let us stop to consider the circumstances in which this debate is taking place and the conclusions to which it should bring us
-1309.47478189

Shortcomings and potential improvements:
    1. Three models are weighed the same (i.e. no weighting at all). Because
    the cost associated with each model is calculated differently, they could
    affect the performance of the decoder significantly simply due to their
    implementations. There should be some smart weighting method to balance
    out their contributions.

    2. Reordering cost is simply based on the distance. This will not scale
    well when translation requires a lot of swapping.

    3. In case of threshold pruning, there's no mechanism to control the stack
    size, which makes the translation time unpredictable. There should be a
    hybrid approach between histogram and threshold pruning to account for
    this.

    4. When a hypothesis is added, it has to be compare with every other
    hypothesis to find out whether there's a combinable hypothesis, hence
    complexity of O(N). We could add a index on large stack based on the
    signature of hypotheses (input sequence, last input word, last output word)
    to reduce search time to O(1).

    5. Due to the implementation using bisect (Python's standard bisection
    algorithm), stack's hypotheses are sorted in increasingly better scored
    order (i.e. worst hypotheses are at the top), so when histogram pruning
    happens, hypotheses[0] is removed and the stack has to shift (size - 1)
    elements. It could be tweaked so that worst hypothesis is at the bottom and
    cost of removing it is O(1).
"""

# Performance Discussion:

# There are two types of hypothesis pruning processes available: Histogram
# pruning and Threshold pruning. Histogram pruning lets the stack of hypotheses
# grow until it reaches a designated MAX_STACK_SIZE. This involves adding every
# hypothesis to the stack and only removing items when it reaches a full state.
# This process is expensive as it allows poor (expensive) hypotheses to be
# added to the stack, those of which will never be used in an actual
# translation. Removing an item is also expensive, because the 'worst'
# translation - that is, Stack[0] is deleted. This means that, at worst,
# MAX_STACK_SIZE-1 elements have to be shuffled 'down' despite the best case
# scenario of only one element being removed.

# The second type of pruning process available is the Threshold pruning
# approach.  Instead of adding every hypothesis into the stack, it determines
# the best (cheapest) costing hypothesis and determines whether or not new
# hypotheses are 'alpha' times more expensive or worse than it. If it is, it is
# never added to the stack, otherwise it is. This method is noticably more
# efficient than the Histogram pruning approach, as poor/expensive hypotheses
# are never added to the stack at all, therefore eliminating the need to delete
# hypotheses and perform the expensive process of reording the stack elements.

# Performance Results:

# Test 1. Execute Histogram pruning on 5 sentences with a stack size of 10.
# Result: 0m54.244s

# Test 2. Execute Histogram pruning on 10 sentences with a stack size of 10.
# Result: 1m59.734s

# Test 3. Execute Histogram pruning on 20 sentences with a stack size of 5.
# Result: 1m12.525s

# Test 4. Execute Threshold pruning on 5 sentences with a stack size of 10.
# Alpha = 0.5 Result: 0m3.301s

# Test 5. Execute Threshold pruning on 10 sentences with a stack size of 10.
# Alpha = 0.5 Result: 0m4.888s

# Test 6. Execute Threshold pruning on 20 sentences with a stack size of 10.
# Alpha = 0.5 Result: 0m7.634s

# Test 7. Execute Threshold pruning on 20 sentences with a stack size of 5.
# Alpha = 0.5 Result: 0m7.579s

# From these tests it can be seen that there is a direct correlation between
# the total duration of processing and the amount of sentences that are given
# as input. Reducing the stack size lowers the processing time, albeit at the
# risk of pruning an hypothesis that could eventually be the cheapest.


from __future__ import division

# custom modules
from ModelExtractor import *
from ReorderingModel import ReorderingModel
from datastructures import Hypothesis, Stack
from utils import get_trans_opts


class Decoder:
    """
    Data structure for representing the phrase decoding process.
    """

    def __init__(self, all_file, e_file, f_file, align_file, max_stack_size):
        """
        Initialise an instance of the Decoder class.

        all_file: Contains all n-gram costs
        e_file: The raw English text input file
        f_file: The raw Foreign (French) text input file
        align_file: The word alignment file
        max_stack_size: The maximum number of hypotheses within a stack
        """
        self.all_file = all_file
        self.english_file = e_file
        self.foreign_file = f_file
        self.alignment_file = align_file
        self.MAX_STACK_SIZE = max_stack_size
        self.lm = SRILangModel()
        self.rm = ReorderingModel()
        self.tm = TranslationModel(self.english_file, self.foreign_file,
                                    self.alignment_file)

    def process_models(self):
        """
        Reads in the 'all' file for the language model and
        extracts the phrases for use in the translation model.
        """
        self.lm.read_lm_file(self.all_file)
        self.tm.extract()

    def get_tm_info(self, sent):
        """
        Gets all the required data from the translation model. A greedy
        approach was taken, where in a sorted list of translation probabilities
        the 'best' probability is chosen and returned.

        A sentence in this case is perceived as containing one or
        more words, ie. a phrase.

        Input: sent - A sentence
        Output: tm_dict - The translation model dictionary
                best_score - The best score for a translation
        """

        if isinstance(sent, list):
            s = ' '.join(sent)

        # just in case a string is passed in by 'accident'
        else:
            s = sent

        tm_dict = self.tm.get_translation_model_prob_e(s)

        # sorted_list[0] = best scoring (entry, prob)
        #   - first entry in sorted trans prob table
        # sorted_list[-1] = worst scoring (entry, prob)
        #   - last entry in sorted trans prob table
        sorted_list = sorted(tm_dict.iteritems(), key=lambda (k, v): (v, k),
                        reverse=True)

        # If there is data to process, get the best score
        # Otherwise there is 'no' best score because an empty list was received
        if sorted_list != []:
            best_score = float(sorted_list[0][1])

        else:
            best_score = None

        return best_score

    def get_future_cost_table(self, sent):
        """
        Used to get the future cost of processing words/phrases from a given
        sentence. This takes into account the translation model's score.  A
        table is generated similar to koehn-06 page 171. Keys are the
        words/phrases, values are their corresponding future costs.

        Future cost ignores reordering model.

        Input: sent - A sentence
        Output: A future cost table

        """
        fc_table = defaultdict(lambda: defaultdict(None))

        for length in range(1, len(sent) + 1):
            for start in range(0, len(sent) + 1 - length):
                end = start + length
                fc_table[start][end] = float('-inf')

                key1 = ' '.join(sent[start:end])
                best_score = self.get_tm_info(key1)
                if best_score is not None:
                    fc_table[start][end] = best_score

                for i in range(start, end - 1):

                    # The cheapest cost estimate for a span is either the
                    # cheapest cost for a translation option or the cheapest
                    # sum of costs for a pair of spans that cover it completely
                    if (fc_table[start][i + 1] +
                        fc_table[i + 1][end]) > fc_table[start][end]:
                        fc_table[start][end] = (fc_table[start][i + 1] + \
                            fc_table[i + 1][end])

        return fc_table

    def decoder_test(self, foreign_file, n_sentences, prune_type,
            alpha=None):
        """
        Test the decoder.

        Input: n_sentences - The number of sentences to parse from the
               foreign file
               foreign_file - a file containing foreign sentences, one per line
               prune_type: "Histogram" or "Threshold"
               alpha: alpha for "Threshold" pruning
        Output: The corresponding English/Foreign sentences and its associated
               processing cost

        """

        f = open(foreign_file, 'r')
        lines = []
        [lines.append(line.split("\n")) \
                for line in f.readlines()[:n_sentences]]

        for i in range(len(lines)):
            print 'Translating phrase %d of %d' % (i + 1, len(lines))
            input_sent = lines[i][0].split(' ')
            fc_table = self.get_future_cost_table(input_sent)

            if alpha is None and prune_type is "Histogram":
                stacks = [Stack(self.MAX_STACK_SIZE, prune_type)
                            for x in range(len(input_sent) + 1)]

            elif prune_type is "Threshold" and alpha is not None:
                stacks = [Stack(self.MAX_STACK_SIZE, prune_type, alpha)
                            for x in range(len(input_sent) + 1)]

            empty_hyp = Hypothesis(None, None, input_sent, fc_table)
            stacks[0].add(empty_hyp)

            for idx, stack in enumerate(stacks):
                for hyp in stack.hypotheses():
                    for trans_opt in get_trans_opts(hyp, self.tm,
                                                     self.rm, self.lm):
                        new_hyp = Hypothesis(hyp, trans_opt)
                        new_stack = stacks[new_hyp.input_len()]
                        new_stack.add(new_hyp)

            last_stack = stacks[-1]
            best_hyp = last_stack.best()

            if best_hyp is not None:
                print ' '.join([y for x in best_hyp.trans['input'] for y in x[2]])
                print ' '.join(best_hyp.trans['output'])
                print best_hyp.trans['score']


if __name__ == '__main__':
    import doctest
    doctest.testmod()
