# Authors:
# Hai Dong Luong (573780) <[hai-ld]>
# Desmond Putra () <[login]>
# Andrew Vadnal (326558) <avadnal>

from __future__ import division
from pprint import pprint

# custom modules
from ModelExtractor import *
from ReorderingModel import ReorderingModel
from datastructures import Hypothesis, Stack
from utils import get_trans_opts

MAX_STACK_SIZE = 10

#############################################
# Model processing
#############################################

lm = SRILangModel()
rm = ReorderingModel()

#read language model file
lm.read_lm_file("source_files/all.lm")

english_file = "source_files/all.lowercased.raw.en"
foreign_file = "source_files/all.lowercased.raw.fr"
alignment_file = "source_files/aligned.grow-diag-final-and"

#run the translation model
tm = TranslationModel(english_file, foreign_file, alignment_file)
tm.extract()


def get_tm_info(sent):
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

    tm_dict = tm.get_translation_model_prob_e(s)

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


def get_lm_cost(sent):
    """
    Used to get the language model cost of a sentence.
    A sentence in this case is perceived as containing one or
    more words, ie. a phrase.

    Input: sent - A sentence
    Output: A probability based on the language model
    """

    if isinstance(sent, list):
        s = ' '.join(sent)

    # just in case a string is passed in by 'accident'
    else:
        s = sent
    return lm.get_language_model_prob(s)


def get_future_cost_table(sent):
    """
    Used to get the future cost of processing words/phrases from
    a given sentence. This takes into account the translation model's score.
    A table is generated similar to koehn-06 page 171. Keys are the
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
            best_score = get_tm_info(key1)
            if best_score is not None:
                fc_table[start][end] = best_score

            for i in range(start, end - 1):
                # The cheapest cost estimate for a span is either the cheapest
                # cost for a translation option or the cheapest sum of costs
                # for a pair of spans that cover it completely

#                print '[', start, '][', i+1, ']', fc_table[start][i+1], '+',
#                print '[', i+1, '][', end, ']', fc_table[i+1][end], '=',
#                   fc_table[start][i+1] + fc_table[i+1][end]
#                print '[start][end]', fc_table[start][end]
#                print '-'*8
                if (fc_table[start][i + 1] +
                    fc_table[i + 1][end]) > fc_table[start][end]:
                    fc_table[start][end] = (fc_table[start][i + 1] +
                        fc_table[i + 1][end])

    return fc_table


if __name__ == '__main__':
    input_sent = 'reprise de la session'.split()
    fc_table = get_future_cost_table(input_sent)

    stacks = [Stack(MAX_STACK_SIZE) for x in range(len(input_sent) + 1)]

    empty_hyp = Hypothesis(None, None, input_sent, fc_table)
    stacks[0].add(empty_hyp)

    for idx, stack in enumerate(stacks):
        i = 0
        for hyp in stack.hypotheses():
            for trans_opt in get_trans_opts(hyp, tm, rm):
                #print trans_opt.input_phrase
                new_hyp = Hypothesis(hyp, trans_opt)
                #print 'idx', idx
                #print 'hyp', hyp
                #print 'new_hyp', hyp
                new_stack = stacks[new_hyp.input_len()]
                new_stack.add(new_hyp)

    #for i in stacks:
    #    print i.hyps[-1]
    last_stack = stacks[-1]
    best_hyp = last_stack.best()
    translation = best_hyp.trans['output']

    if best_hyp is not None:
        print best_hyp.trans['input']
        print best_hyp.trans['output']
        print best_hyp.trans['score']
