# Authors: 
# Hai Dong Luong () <[hai-ld]>
# Desmond Putra () <[login]>
# Andrew Vadnal (326558) <avadnal>

from __future__ import division
from pprint import pprint
from sys import maxint as INFINITY
from ModelExtractor import *

import bisect

MAX_STACK_SIZE = 100

#############################################
# Class declarations
#############################################

class ReorderingModel:
    def __init__(self):
        pass

    def cost(self, prev_phrase, next_phrase):
        return 1.0

class TranslationModel:
    def __init__(self):
        pass

    def translate(self, phrase):
        return {
            'trans-0': 0.1,
            'trans-1': 0.5,
        }

class Hypothesis:
    def __init__(self, hyp, trans_opt):
        """Create a hypothesis expanding on another hypothesis by a translation option."""
        #self.prev = hyp # pointer to previous hypothesis
        #self.next = [] # pointers to next hypotheses
        if hyp is not None:
            #hyp.next.append(self)
            self.trans = {}
            self.trans['input'] = hyp.trans['input'] + [(trans_opt.i_start, trans_opt.i_end, trans_opt.input_phrase)]
            self.trans['output'] = hyp.trans['output'] + [trans_opt.output_phrase]
            self.trans['score'] = hyp.trans['score'] * trans_opt.score
        else: # create an empty hypothesis
            self.trans = {
                'input': [],
                'output': [],
                'score': 1.0,
            }

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
        return self.trans['score'] < other.trans['score']

    def __le__(self, other):
        return self.trans['score'] <= other.trans['score']

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
        self.hyps = [] # list of hypotheses in ascending order

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
                del self.hyps[j-i]

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
        return self.hyps[-1]

class TranslationOption:
    def __init__(self, i_start, i_end, input_phrase, output_phrase, score):
        """Create a translation option.
        i_start: start position of input phrase
        i_end: end position of input phrase
        input_phrase: input phase as a list of words
        output_phrase: output phrase as a string
        score: the score assigned to this translation
        """
        self.i_start = i_start
        self.i_end = i_end
        self.input_phrase = input_phrase
        self.output_phrase = output_phrase
        self.score = score


#############################################
# Model processing
#############################################

LM = SRILangModel()
TM = TranslationModel()
RM = ReorderingModel()

#read language model file
LM.read_lm_file("source_files/all.lm")

#find the score (log10)
# output_lm = LM.get_language_model_prob("accommodated")
# print output_lm

#translation model
english_file = "source_files/all.lowercased.raw.en"
foreign_file = "source_files/all.lowercased.raw.fr"
alignment_file = "source_files/aligned.grow-diag-final-and"

#run the translation model
# tm = TranslationModel(english_file, foreign_file, alignment_file)
# tm.extract()

#find list of translations
# output_tm = tm.get_translation_model_prob("en")

# #find the score (log10) sort by highest score
# for key, value in sorted(output_tm.iteritems(), key=lambda (k,v): (v,k), reverse=True):
#     print key, value

#############################################
# Utility functions
#############################################

def get_all_phrases(sentence):
    """Get all phrases in a sentence.
    sentence: a sentence given as a list of words
    """
    for i in range(len(sentence)):
        for j in range(i + 1, len(sentence) + 1):
            yield sentence[i:j]

def get_trans_opts(input_sent, hyp):
    """Get all translation options a hypothesis could be expanded upon.
    input_sent: input sentence given as a list of words
    hyp: the hypothesis to be expanded
    """
    untrans = get_untranslated_words(input_sent, hyp)

    for part in get_consecutive_parts(untrans):
        for phrase in get_all_phrases(part):
            reordering_score = 1.0 #reordering_model(hyp.trans['input'][-1], phrase)
            for translation in get_translations(phrase):
                translation.score *= reordering_score
                yield translation

def get_untranslated_words(input_sent, hyp):
    """Get words untranslated by a hypothesis.
    input_sent: input sentence given as a list of words
    hyp: a hypothesis translating the input sentence

    Returns a list of tuples in which second elements are untranslated words
    and first elements are positions of the words in input sentence.
    """
    input_sent = dict(enumerate(input_sent))
    for i in hyp.trans['input']:
        for j in range(i[0], i[1] + 1):
            del input_sent[j]

    return input_sent.items()

def get_consecutive_parts(input_sent):
    """Get consecutive parts in a non-consecutive sentence.
    input_sent: input sentence given as a list of tuples in which second elements are untranslated words
    and first elements are positions of the words in input sentence.

    Return a list of consecutive parts, each of which is a list of words
    """
    consecutive_parts = []
    prev_idx = None
    part = []
    for idx, word in input_sent:
        if prev_idx is None:
            part.append((idx, word))
        else:
            if idx == prev_idx + 1:
                part.append((idx, word))
            else:
                consecutive_parts.append(part)
                part = [(idx, word)]
        prev_idx = idx
    consecutive_parts.append(part)
    return consecutive_parts

def get_translations(phrase):
    for translation, score in TM.translate(phrase).iteritems():
        t = TranslationOption(phrase[0][0], phrase[-1][0], [p[1] for p in phrase], translation, score)
        yield t

def get_possible_phrases_test():
    input_sent = dict(enumerate('who is but a function of what'.split()))
    del input_sent[1]
    del input_sent[6]

    for p in get_possible_phrases(input_sent.items()):
        print p

def get_untranslated_parts_test():
    input_sent = 'U o M'.split()
    h0 = Hypothesis(None, None)
    trans_opt = TranslationOption(1, 1, ['o', 'M'], ['z', 'z'], .2)
    h = Hypothesis(h0, trans_opt)

    print get_untranslated_parts(input_sent, h)

def recombine(stacks):
    pass

def prune(stacks):
    pass

# Base pruning score on future cost (need to estimate) too
# Gets added to partial probability score
# Future cost ignores reordering model 

# The cheapest cost estimate for a span is either the cheapest cost for a 
# translation option or the cheapest sum of costs for a pair of spans that cover
# it completely

# Take into account translation model and language model probabilities
# Set the cost required to process from start to end. Val can be used to set cost to infinity
def get_cost(start, end, val=None):
    
    if val is not None:
        the_cost = val

    # need to get this stuff from the language model
    else:
        the_cost = translation_prob(start) + translation_prob(end)

    return the_cost

# Based on current costs, we want to estimate the future costs
def get_future_cost(n_words):
    for length in range(1, n):
        for start in range(1, n_words+1-length):
            end = start + length

            #initialise the cost from start->end to be infinity
            start_to_end_cost = get_cost(start, end, INFINITY)

            # If a translation option for a cost estimate exists
            if trans_opt_cost_estimate:
                start_to_end_cost = cost_estimate

            for i in range(start, end-1):
                partial_cost = get_cost(start, i) + get_cost(i+1, end)
                if partial_cost < get_cost(start_end):
                    start_to_end_cost = partial_cost

    return start_to_end_cost

def pruning_histogram(stack, pruning_limit):
    """
    Keep a maximum of n hypotheses in the stack
    Number of hypothesis expansions =
        Max stack size * number of translation options * length input sent

    Advantage: Improvement from exponential cost
    Disadvantage: Not the best translation according to the Model

    Use this to demonstrate a less optimal way of pruning

    Input: stack - A stack of hypotheses
    Input: pruning_limit - A limit of how many hypotheses to prune
    Output: A pruned stack

    """
    # Check the stack size
    if len(stack) > MAX_STACK_SIZE:

        # Remove the specified amount of hypotheses from the stack and return stack
        [stack.pop(0) for i in range(pruning_limit)]
        return stack
    
    # Just return the original stack as it does not need to be pruned    
    else:
        return stack

def pruning_threshold(alpha, stack):
    """
    Check if a hypothesis score is 'alpha' times worse than the best score
    If this is the case, prune it from the stack

    Input: alpha - A threshold value
    Input: stack - A stack of hypotheses
    Output: A pruned stack

    """
    num_words = len(stack.hypotheses)
    scores = []

    # Find the highest scoring hypothesis in the stack
    for hyp in stack:

        scores.append(hyp['score'])

    best_score = max(scores)

    # Prune the stack
    for hyp in stack:

        # Get the probability score of the hypothesis
        prob_score = hyp['score']
        
        # Get the future cost of processing the next n hypotheses
        future_cost = self.get_future_cost(num_words)

        # If the score of a hypothesis is 'threshold' times worse than best, prune it
        if best_score / (prob_score + future_cost) < alpha:  
            stack.remove(hyp)

    return stack



input_sent = 'Uni of Melb'.split()

stacks = [Stack(MAX_STACK_SIZE) for x in range(len(input_sent) + 1)]

empty_hyp = Hypothesis(None, None)
stacks[0].add(empty_hyp)

for idx, stack in enumerate(stacks):
    i = 0
    for hyp in stack.hypotheses():
        for trans_opt in get_trans_opts(input_sent, hyp):
#            print trans_opt.input_phrase
            new_hyp = Hypothesis(hyp, trans_opt)
#            print 'idx', idx
#            print 'hyp', hyp
#            print 'new_hyp', hyp
            new_stack = stacks[new_hyp.input_len()]
            new_stack.add(new_hyp)
            recombine(stacks)
            prune(stacks)

last_stack = stacks[-1]
best_hyp = last_stack.best()
#translation = best_hyp.trans['output']

print best_hyp.trans['input']
print best_hyp.trans['output']
print best_hyp.trans['score']

for stack in stacks:
    print len(stack.hypotheses())
    pprint(stack.hypotheses())
