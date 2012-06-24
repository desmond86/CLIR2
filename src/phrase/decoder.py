# Authors: 
# Hai Dong Luong (573780) <[hai-ld]>
# Desmond Putra () <[login]>
# Andrew Vadnal (326558) <avadnal>

from __future__ import division
from pprint import pprint

# use float('-inf') instead
# from sys import maxint as INFINITY
from ModelExtractor import *
import bisect

MAX_STACK_SIZE = 10

#############################################
# Class declarations
#############################################

class ReorderingModel:
    def __init__(self):
        pass

    def score(self, prev_phrase, next_phrase):
        """Calculate reordering cost based on distance between phrases."""
        dist = next_phrase[0][0] - prev_phrase[1] - 1
        return -(abs(dist))


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
        else: # create an empty hypothesis
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

        if len(self.hyps) > MAX_STACK_SIZE:
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
    the 'best' probability is chosen and returned. The dictionary itself is
    returned purely as a convenience.

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
    sorted_list = sorted(tm_dict.iteritems(), key=lambda (k,v): (v,k),
                    reverse=True)
    
    # If there is data to process, get the best score
    # Otherwise there is 'no' best score because an empty list was received
    if sorted_list != []:
        best_score = float(sorted_list[0][1])

    else:
        best_score = None
    
    return tm_dict, best_score

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

# Based on current costs, we want to estimate the future costs
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

    for length in range(1,len(sent)+1):
        for start in range(0, len(sent)+1-length):
            end = start + length
            fc_table[start][end] = float('-inf')

            key1 = ' '.join(sent[start:end])
            trans_prob, best_score = get_tm_info(key1)#
            if best_score is not None:
                fc_table[start][end] = best_score
            
            for i in range(start, end-1):
                # The cheapest cost estimate for a span is either the cheapest
                # cost for a translation option or the cheapest sum of costs
                # for a pair of spans that cover it completely

#                print '[', start, '][', i+1, ']', fc_table[start][i+1], '+',
#                print '[', i+1, '][', end, ']', fc_table[i+1][end], '=',
#                   fc_table[start][i+1] + fc_table[i+1][end]
#                print '[start][end]', fc_table[start][end]
#                print '-'*8
                if (fc_table[start][i+1] +
                    fc_table[i+1][end]) > fc_table[start][end]:
                    fc_table[start][end] = (fc_table[start][i+1] + 
                        fc_table[i+1][end])

        
    return fc_table

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

def get_trans_opts(hyp):
    """Get all translation options a hypothesis could be expanded upon.
    hyp: the hypothesis to be expanded
    """
    untrans = get_untranslated_words(hyp)

    for part in get_consecutive_parts(untrans):
        for phrase in get_all_phrases(part):
            try:
                reordering_score = rm.score(hyp.trans['input'][-1], phrase)
            except IndexError:
                reordering_score = 0.0
            for translation in get_translations(phrase):
                translation.score += reordering_score
                yield translation

def get_untranslated_words(hyp):
    """Get words untranslated by a hypothesis.

    Returns a list of tuples in which second elements are untranslated words
    and first elements are positions of the words in input sentence.
    """
    input_sent = dict(enumerate(hyp.input_sent))
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
    phrase_words = [p[1] for p in phrase]
    #print len(tm.get_translation_model_prob_f(' '.join(phrase_words))), ' '.join(phrase_words)
    for translation, score in tm.get_translation_model_prob_f(' '.join(phrase_words)).iteritems():
        t = TranslationOption(phrase[0][0], phrase[-1][0], phrase_words, translation, score)
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

input_sent = 'the tourism initative addresses this for the first time'.split() #'reprise de la session'.split()
fc_table = get_future_cost_table(input_sent)
pprint(dict(fc_table))

stacks = [Stack(MAX_STACK_SIZE) for x in range(len(input_sent) + 1)]

empty_hyp = Hypothesis(None, None, input_sent, fc_table)
stacks[0].add(empty_hyp)

for idx, stack in enumerate(stacks):
    i = 0
    for hyp in stack.hypotheses():
        for trans_opt in get_trans_opts(hyp):
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

if best_hyp is not None:
    print best_hyp.trans['input']
    print best_hyp.trans['output']
    print best_hyp.trans['score']

#for stack in stacks:
#    print len(stack.hypotheses())
#    pprint(stack.hypotheses())
