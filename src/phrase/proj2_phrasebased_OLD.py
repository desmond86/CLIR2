#Authors:
# Authors: 
# Hai ..? <[login]>
# Desmond Putra <[login]>
# Andrew Vadnal <avadnal>

from nltk.align import AlignedSent

class PhraseBasedModel:
    """
    Phrase Based Model class - containing all functionality outlined in
    the Koehn algorithm - Chapter 5. Figure 5.5 - page 133.
    The paper can be found at: 
    http://langtech.github.com/clir/materials/koehn-05.pdf
    """

    def __init__(self, aligned_sent):
        """
        Class constructor. 
        Input: aligned_sent
            Object of type NLTK.align.AlignedSent. This object
            contains an english sentence, a foreign sentence, and a
            word alignment list. The word alignment list is a list word-level
            pair mappings. The english and foreign sentences can be obtained
            by calling the methods 'words' and 'mots' on the AlignedSent object
            respectively. The word alignments can be accessed by calling the
            'alignment' method on the AlignedSent object. 
        Output:
            None
        """

        # Get the corresponding sentences and alignment
        self.eng_sent = aligned_sent.words
        self.for_sent = aligned_sent.mots
        self.word_alignments = aligned_sent.alignment

        # eng_list will store the output of the extract function
        self.eng_list = list()

        # phrase_pairs will store the final english-foreign based pairs
        # the phrase pairs will be stored as so:
        # [(['e1','e2',...,'en'],['f1','f2',...,'fn']),...,] 
        # where e1..en and f1..fn make up the resultant phrase pair
        self.phrase_pairs = list()

    def extract(self, f_start, f_end, e_start, e_end):
        """
        The extraction of word alignments for sentence pairs.
        Input: f_start
            First word of minimal phrase of aligned foreign words (lower bound)
        Input: f_end
            Last word of minimal phrase of aligned foreign words (upper bound)
        Input: e_start
            The first word in an English phrase (lower bound)
        Input: e_end
            The last word in an English phrase (upper bound)
        Output:
            An extracted phrase pair
        """

        # If there are no alignment points
        if f_end == -1: 
            return set()

        # Conditions outlined on page 131 of the Koehn paper
        # A phrase pair (f,e) is deemed consistent with an alignment A
        # if all words f1,..,fn in f that have alignment points in A 
        # have these with words e1,..,en in e and vice versa.

        for e,f in self.word_alignments:
            # Phrase pair is consistent
            if (e_start <= e <= e_end
                and
                f_start <= f <= f_end
                ):
                continue

            # Phrase pair is consistent
            elif (e_start > e or
                  e_end < e) and (f_start > f or f_end < f):
                continue

            # Phrase pair is inconsistent
            elif (f_start > f or
                  f_end < f) and (e_start <= e <= e_end):
                return set()

            # Phrase pair is inconsistent    
            elif (e_start > e or
                  e_end < e and f_start <= f <= f_end):
                return set()
                     
        # add phrase pairs (incl. additional unaligned f)
        fs = f_start

        # keep track of the values at the foreign index
        f_index = []
        for e,f in self.word_alignments:
             f_index.append(f)

        # was not sure what Koehn meant by 'until fs aligned'
        while True: 
            fe = f_end

            # was not sure what Koehn meant by 'until fe aligned'
            while True:

                # add phrase pair to eng_list
                self.eng_list.append(((e_start, e_end), (fs, fe)))
                fe += 1

                # was fe located in the word alignments specified?
                if fe in f_index or fe > len(self.for_sent)-1:
                    break

            fs -= 1
            # was fs located in the word alignments specified?
            if fs in f_index or fs < 0:
                break

        return self.eng_list     

    def execute(self, eng_sent, for_sent):
        """
        Execution of the phrase pair extraction algorithm
        Input: eng_sent
            An english sentence

        Input: for_sent
            A foreign sentence

        Output: 
            None
        """

        # range starting at 0 because alignment starts at (0,0) not (1,1)
        for e_start in range(0, len(eng_sent)):
            for e_end in range(e_start, len(eng_sent)):
                # find the minimally matching foreign phrase

                # len-1 because alignment starts at (0,0) not (1,1)
                f_start = len(for_sent)-1
                f_end = 0

                # find the min and max phrase of aligned words in 
                # english phrase e, spanning from e_start to e_end
                for e,f in self.word_alignments:
                    if e_start <= e <= e_end: 
                        f_start = min(f, f_start)
                        f_end = max(f, f_end)

                extraction = self.extract(f_start, f_end, e_start, e_end)

                for i in extraction:

                    # this check prevents duplicate pairs from being stored
                    if i not in self.phrase_pairs:
                        self.phrase_pairs.append(i)

        # print out the phrase pairs that have been collected                
        self.print_phrase_pairs(self.phrase_pairs)

# Base pruning score on future cost (need to estimate) too
# Gets added to partial probability score
# Future cost ignores reordering model 

# The cheapest cost estimate for a span is either the cheapest cost for a 
# translation option or the cheapest sum of costs for a pair of spans that cover
# it completely

# Take into account translation model and language model probabilities into account
# Set the cost required to process from start to end. Val can be used to set cost to infinity
def get_cost(self, start, end, val=None):
    
    # 
    if val is not None:
        the_cost = val

    # need to get this stuff from the language model
    else:
        the_cost = translation_prob(start) + translation_prob(end)

    return the_cost

# Based on current costs, we want to estimate the future costs
def get_future_cost(self, n_words):
    for length in range(1, n):
        for start in range(1, n_words+1-length):
            end = start + length
            #initialise the cost from start->end to be infinity
            # FROM SYS IMPORT MAXINT AS INFINITY

            start_to_end_cost = self.get_cost(start, end, INFINITY)

            # If a translation option for a cost estimate exists
            if trans_opt_cost_estimate:
                start_to_end_cost = cost_estimate

            for i in range(start, end-1):
                partial_cost = get_cost(start, i) + get_cost(i+1, end)
                if partial_cost < get_cost(start_end):
                    start_to_end_cost = partial_cost

    return start_to_end_cost

def pruning_histogram(self, stack, pruning_limit):
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

def pruning_threshold(self, alpha, stack):
    """
    Check if a hypothesis score is 'alpha' times worse than the best score
    If this is the case, prune it from the stack

    Input: alpha - A threshold value
    Input: stack - A stack of hypotheses
    Output: A pruned stack

    """
    num_words = NUMBER_OF_WORDS(self.sentence)
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
    
     
class Hypothesis:

    def __init__(self, hyp=None, en_phrase=None, fr_phrase=None, score=None):
        """
        Class constructor.
        Input:
            An hypothesis. This will be used later to expand on earlier
            hypotheses in order to create a new hypothesis
            English phrase. Pre-generated from the phrase extraction algorithm
            Foreign phrase. Taken from the input sentence
            Score. The partial score of the hypothesis
        Output:
            None
        """

        # When the first hypothesis is generated, we need to generate empty
        # values which will then be extended upon when new hypotheses are
        # created
        
        if hyp is None:
            self._trans = []
            self._score = 0
            self._pointer = super().__init__() #init this class 

        else:
            
            self._trans = hyp._trans + [(en_phrase, fr_phrase, score)]
            self._score = hyp._score + score #the partial score
            self._ptr = hyp
            self._en_phrase = en_phrase
            self._fr_phrase = fr_phrase

    def extend(self, en_phrase, fr_phrase, score):
        return Hypothesis(self, FtoE_Trans(fr_phrase)[0], fr_phrase, FtoE_Trans(fr_phrase)[1])

 
class FtoE_TransTable:
    
    def __init__(self, phrase_table):
        self._phrase_table = phrase_table

    def add(self, phrase_pairs):
        
    def trans(self, f):
        return (e, score)


trans_table = FtoE_TransTable(None)
for e, f, word_alignments in data:
    aligned_sent = nltk_align.AlignedSent(e, f, word_alignments)
    pbm = PhraseBasedModel(aligned_sent)
    pbm.execute(pbm.eng_sent, pbm.for_sent)
    trans_table.add(pbm.phrase_pairs)
