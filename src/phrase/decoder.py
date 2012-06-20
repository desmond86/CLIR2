# Authors: 
# Hai Dong Leong () <[login]>
# Desmond Putra () <[login]>
# Andrew Vadnal (326558) <avadnal>

from __future__ import division
from pprint import pprint
from sys import maxint as INFINITY
from ModelExtractor import *

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
        self.prev = hyp
        self.next = []
        if hyp is not None:
            hyp.next.append(self)
            self.trans = {}
            self.trans['input'] = hyp.trans['input'] + [(trans_opt.i_start, trans_opt.i_end, trans_opt.input_phrase)]
            self.trans['output'] = hyp.trans['output'] + [trans_opt.output_phrase]
            self.trans['score'] = hyp.trans['score'] * trans_opt.score
        else:
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

    def __str__(self):
        return str(self.trans)

class Stack:
    def __init__(self, size):
        self.size = size
        self.hyps = []

    def add(self, hyp):
        self.hyps.append(hyp)

    def hypotheses(self):
        return self.hyps

    def best(self):
        return self.hyps[-1]

class TranslationOption:
    def __init__(self, i_start, i_end, input_phrase, output_phrase, score):
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


#def get_all_phrases(sentence):
#    if len(sentence) == 1:
#        yield [sentence]
#    else:
#        for i in range(1, len(sentence) + 1):
#            if i == len(sentence):
#                yield [sentence]
#            pre = [sentence[:i]]
#            for phrase in get_all_phrases(sentence[i:]):
#                yield pre + phrase

#############################################
# Utility functions
#############################################

def get_all_phrases(sentence):
    for i in range(len(sentence)):
        for j in range(i + 1, len(sentence) + 1):
            yield sentence[i:j]

def get_trans_opts(input_sent, hyp):
    untrans = get_untranslated_parts(input_sent, hyp)
    possible_phrases = get_possible_phrases(untrans)

    for phrase in possible_phrases:
        score = 1.0 #reordering_model(hyp.trans['input'][-1], phrase)
        for translation in get_possible_translations(phrase):
            yield translation

def get_untranslated_parts(input_sent, hyp):
    input_sent = dict(enumerate(input_sent))
    for i in hyp.trans['input']:
        for j in range(i[0], i[1] + 1):
            del input_sent[j]

    return input_sent.items()

def get_possible_phrases(input_sent):
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

    for part in consecutive_parts:
        for phrase in get_all_phrases(part):
            yield phrase

def get_possible_translations(phrase):
    #print phrase
    for translation, score in TM.translate(phrase).iteritems():
        t = TranslationOption(phrase[0][0], phrase[-1][0], [p[1] for p in phrase], translation, score)
#        print t.i_start, t.i_end
#        print t.input_phrase
#        print t.output_phrase
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