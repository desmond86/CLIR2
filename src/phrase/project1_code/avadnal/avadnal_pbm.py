#Andrew Vadnal
#326558
#24/4/2012
#Project 1 - Phrase Based Model (implementation)

"""
>>> e = "michael assumes that he will stay in the house"
>>> e = e.split()
>>> f = "michael geht davon aus , dass er im haus bleibt"
>>> f = f.split()
>>> word_alignments = [(0,0),(1,1),(1,2),(1,3),(2,5),(3,6),(4,9),
...                   (5,9),(6,7),(7,7),(8,8)]
>>> aligned_sent = nltk_align.AlignedSent(e, f, word_alignments)
>>> pbm = PhraseBasedModel(aligned_sent)
>>> pbm.execute(pbm.eng_sent, pbm.for_sent)
michael -- michael
michael assumes -- michael geht davon aus
michael assumes -- michael geht davon aus ,
michael assumes that -- michael geht davon aus , dass
michael assumes that he -- michael geht davon aus , dass er
michael assumes that he will stay in the house -- michael geht davon aus , dass er im haus bleibt
assumes -- geht davon aus
assumes -- geht davon aus ,
assumes that -- geht davon aus , dass
assumes that he -- geht davon aus , dass er
assumes that he will stay in the house -- geht davon aus , dass er im haus bleibt
that -- dass
that -- , dass
that he -- dass er
that he -- , dass er
that he will stay in the house -- dass er im haus bleibt
that he will stay in the house -- , dass er im haus bleibt
he -- er
he will stay in the house -- er im haus bleibt
will stay -- bleibt
will stay in the house -- im haus bleibt
in the -- im
in the house -- im haus
house -- haus
"""

from __future__ import division
from collections import defaultdict

#Had some issues downloading the align package at uni due to the proxy settings
#I've just directly grabbed the source code from the NLTK site and included
#it in this project as a quick fix.

# Discussion:
# ==============================================================================
# The implemented algorithm was essentially the 'bare bones' approach to phrase
# processing, whereby only the phrase pairs are extracted out of an English and
# foreign sentence and displayed back to the user. The next step for this
# algorithm is to estimate translation probabilities (by using a relative
# frequency) based on the generated phrase pairs.

# Given more time on this project, I would extend current model implementing the
# the phrase extraction algorithm to allow  for a log-linear model to be used. The
# algorithm that was implemented in this project is naive, in that it merely
# gathers phrase pairs based on word alignments, resulting in output that may not
# be considered 'good English'. This is because the current translation model of a
# phrase doesn't depend on the phrases that surround it. Practically speaking,
# this approach would not be applicable in a functional CLIR system, as users only
# desire a sentence which is syntactically correct and that the meaning is
# discernible semantically. One approach I wanted to explore but unfortunately met
# my time quota was a lexical-weighting approach which revolves around computing
# the likelihood of words appearing besides other words, in turn producing more
# desirable phrase pairs. I feel that this approach would generate, especially 
# over a larger data set than the one used to complete this assignment, a much
# more realistic set of phrase pairs - namely English sentences that are complete
# grammatically sound.
# ==============================================================================

import nltk_align

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

    def print_phrase_pairs(self, phrase_pairs):
        """
        Input:
            List of phrase pairs in form:
            [(['e1','e2',...,'en'],['f1','f2',...,'fn']),...,] 

        Output:
            The extracted phrase pairs are printed to screen
        """

        e = []
        f = []

        for i in self.phrase_pairs:

            #given pair ((a,b), (c,d))    
            a = i[0][0]
            b = i[0][1]
            c = i[1][0]
            d = i[1][1]

            e.append(self.eng_sent[a:b+1])
            f.append(self.for_sent[c:d+1])
        
        for i in zip(e,f):
             print "%s -- %s" % (" ".join(i[0]), " ".join(i[1]))

# e = "michael assumes that he will stay in the house".split()
# f = "michael geht davon aus , dass er im haus bleibt".split()

# word_alignments = [(0,0),(1,1),(1,2),(1,3),(2,5),(3,6),(4,9),
#                   (5,9),(6,7),(7,7),(8,8)]

# aligned_sent = nltk_align.AlignedSent(e, f, word_alignments)

# pbm = PhraseBasedModel(aligned_sent)
# pbm.execute(pbm.eng_sent, pbm.for_sent)

if __name__ == "__main__":
    import doctest
    doctest.testmod()