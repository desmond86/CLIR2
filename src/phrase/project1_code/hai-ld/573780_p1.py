# Luong Dong Hai
# 573780
# 2012-04-24
"""
Implement phrase-based model as described in Koehn-05.

1. Description of the model
Translation models based on words have two major drawbacks. Firstly, words may
actually not be the smallest units of languages, as one word in a language
could be translated in to two or more words in another language. Secondly,
word-based models do not leverage surrounding context of source language, which
is an important translation clue.

Phrase-based models overcome these problems by capturing translated phrases
during training and reusing them when dealing with unseen data. Phrases solve
one-to-many and many-to-many mapping problems, as well as capture the context
of source as well as translated text.

Koehn-05's phrase-based model defines best English translation for a foreign
sentence as:
    e_best = argmax_e product(p(f|e)) * d * product(p_LM(e_i|e_1..e_i-1))
in which f, e is foreign and English phrases, respectively; d is reordering
cost and product(p_LM(e_i|e_1..e_i-1)) is language model score for English
translation

2. Implementation
There are 4 components for this model: a phrase translation table extractor, a
language model, a distance-based cost function and a translator utilizing them
all.
- Phrase translation table are extracted from English and foreign sentences,
  given their word alignments, as described in Koehn-05, p130..136. This model
  doesn't implement word alignments extraction algorithm. They're explicitly
  given.
>>> e = 'michael assumes that he will stay in the house'.split()
>>> f = 'michael geht davon aus , dass er im haus bleibt'.split()
>>> a = [
...         (0, 0),
...         (1, 1),
...         (1, 2),
...         (1, 3),
...         (2, 5),
...         (3, 6),
...         (4, 9),
...         (5, 9),
...         (6, 7),
...         (7, 7),
...         (8, 8)
... ]
>>> e1 = 'in the old house'.split()
>>> f1 = 'im alt haus'.split()
>>> a1 = [(0, 0), (1, 0), (2, 1), (3, 2)]
>>> extractor = PhraseTableExtractor([(e, f, a), (e1, f1, a1)])
>>> table = extractor.extract()[0]
>>> print extractor # doctest: +ELLIPSIS
Pr(e|f)
f ||| e ||| Pr
alt ||| old ||| 1.0
...
haus ||| house ||| 1.0
geht davon aus , dass ||| assumes that ||| 1.0
geht davon aus , dass er ||| assumes that he ||| 1.0
er im haus bleibt ||| he will stay in the house ||| 1.0
dass er im haus bleibt ||| that he will stay in the house ||| 1.0
, dass er im haus bleibt ||| that he will stay in the house ||| 1.0
im alt haus ||| in the old house ||| 1.0
bleibt ||| will stay ||| 1.0
...
Pr(f|e)
e ||| f ||| Pr
michael assumes that ||| michael geht davon aus , dass ||| 1.0
michael assumes ||| michael geht davon aus , ||| 0.5
assumes that he ||| geht davon aus , dass er ||| 1.0
that ||| dass ||| 0.5
will stay ||| bleibt ||| 1.0
that he will stay in the house ||| , dass er im haus bleibt ||| 0.5
...
old house ||| alt haus ||| 1.0
house ||| haus ||| 1.0
michael assumes ||| michael geht davon aus ||| 0.5
...

- This module use a trigram language model, based on conditional frequency,
  which means that it doesn't account for unseen events. This decision is due
  to a technical difficulty with SimpleGoodTuringProbDist (explained in
  implementation details) and should be fixed.
>>> lm = TrigramLM([e, e1])

- Reordering cost is calculated as N^|x|, where x is the total distance input
  phrases have to move during reordering, and N is in [0,1], as described in
  Koehn-05, p129
- Translator works as follows:
    - Generates all possible phrase segmentations. In the worst case, sentence
      of length N would have 2^(N-1)
    - For each segmentation, generates all possible ordering. If there are N
      phrase, the number of ordering is N!
    - For each ordering, generate all possible phrase translation.
    - For each translation, calculate its probability based on given trigram
      language model

>>> translator = Translator(table, d, lm)
>>> print translator.translate(
...         'michael geht davon aus , dass er im alt haus bleibt'.split())
['michael', 'assumes', 'that', 'he', 'will', 'stay', 'in', 'the', 'old', \
'house']

3. Possible improvements
- Every segmentation is assumed to be equally likely. A parameter may be added
  to bias the translation model toward longer (less) or shorter (more) phrases.
- 3 components used by the translator are given equal weight of 1. The
  translator should be able to give them different weight, probably by using
  log-linear model (Koehn-05, p136)
- Generated translations have many duplicates due to the fact that different
  segmentations can be translated into the same sentence (e.g. [A, BC, D] and
  [A, B, C, D] is actually the same). Only the translation with highest score
  should be retained.
- About 2^(L-1) * M! * N translations are generated. That's a lot(!!!), with
  many duplicates. There should be some heuristics to detect duplication early.
"""
from __future__ import division
from collections import defaultdict
from pprint import pprint
from StringIO import StringIO
from nltk import ConditionalFreqDist, ConditionalProbDist, \
    SimpleGoodTuringProbDist
from nltk.util import ngrams


class PhraseExtractor:
    """Extract phrase pairs from English and foreign sentences given their
    word alignment.

    Alignment is a list of tuples whose first items are indices of English
    words and second items are indices of foreign words.
    """
    def __init__(self, e, f, alignment):
        self.e = e
        self.f = f
        self.alignment = alignment
        self.phrase_pairs = []

    def phrase_extract(self):
        """Extract and return phrase pairs.

        A phrase pair is a pair of tuples. First one is start and end index of
        English phrase, second one is start and end index of foreign phrase."""
        e, f, alignment = self.e, self.f, self.alignment

        # considers all possible English phrases
        for e_start in range(len(e)):
            for e_end in range(e_start, len(e)):
                f_start, f_end = len(f), 0
                # find minimally aligned foreign phrase
                for e_i, f_i in alignment:
                    if e_start <= e_i <= e_end:
                        f_start = min(f_i, f_start)
                        f_end = max(f_i, f_end)

                self.phrase_pairs.extend(
                    self.extract(f_start, f_end, e_start, e_end)
                )
        return self.phrase_pairs

    def extract(self, f_start, f_end, e_start, e_end):
        """Extract and return phrase pairs with fixed English phrase.

        English phrase starts from e_start to e_end,
        foreign phrases include minimally aligned phrase f_start..f_end and
        phrases extended to neighboring unaligned words.
        """
        # consistency check
        for e_i, f_i in self.alignment:
            if f_start <= f_i <= f_end and (e_i < e_start or e_i > e_end):
                return []

        phrase_pairs = []
        f_s = f_start
        while True:
            f_e = f_end
            while True:
                phrase_pair = (
                        (e_start,   e_end),
                        (f_s,       f_e)
                )
                phrase_pairs.append(phrase_pair)
                f_e += 1
                if self.is_aligned(f_e):
                    break
            f_s -= 1
            if self.is_aligned(f_s):
                break
        return phrase_pairs

    def is_aligned(self, f):
        """Check if a foreign word is aligned"""
        if f < 0 or f >= len(self.f):
            return True
        for ae, af in self.alignment:
            if f == af:
                return True
        return False


class PhraseTableExtractor:
    """Extract phrase translation table of from word alignments.

    Alignments is a list of tuples whose first items are English sentences,
    second are foreign sentences and third are their word alignments."""
    def __init__(self, alignments):
        self.alignments = alignments
        self.prob_fe = self.prob_ef = None

    def extract(self):
        """Extract phrase translation table."""
        alignments = self.alignments
        count_ef = defaultdict(lambda: defaultdict(int))
        count_e = defaultdict(int)
        count_f = defaultdict(int)
        for e, f, alignment in alignments:
            extractor = PhraseExtractor(e, f, alignment)
            phrase_pairs = extractor.phrase_extract()
            # count phrase occurrences
            for (e_start, e_end), (f_start, f_end) in phrase_pairs:
                e_phrase = ' '.join(e[e_start:e_end + 1])
                f_phrase = ' '.join(f[f_start:f_end + 1])
                count_ef[e_phrase][f_phrase] += 1
                count_e[e_phrase] += 1
                count_f[f_phrase] += 1
        # Pr(f|e)
        prob_fe = defaultdict(lambda: defaultdict(float))
        # Pr(e|f)
        prob_ef = defaultdict(lambda: defaultdict(float))
        for e_phrase in count_ef:
            for f_phrase in count_ef[e_phrase]:
                prob_fe[f_phrase][e_phrase] = \
                    count_ef[e_phrase][f_phrase] / count_e[e_phrase]

                prob_ef[e_phrase][f_phrase] = \
                    count_ef[e_phrase][f_phrase] / count_f[f_phrase]

        self.prob_fe, self.prob_ef = prob_fe, prob_ef
        return prob_fe, prob_ef

    def __str__(self):
        s = StringIO()
        print >>s, 'Pr(e|f)'
        print >>s, 'f ||| e ||| Pr'
        for e in self.prob_ef:
            for f in self.prob_ef[e]:
                print >>s, f, '|||', e, '|||', self.prob_ef[e][f]

        print >>s, 'Pr(f|e)'
        print >>s, 'e ||| f ||| Pr'
        for f in self.prob_fe:
            for e in self.prob_fe[f]:
                print >>s, e, '|||', f, '|||', self.prob_fe[f][e]
        return s.getvalue()


class Translator:
    """Translate using phrase-based model."""
    def __init__(self, table, reorder_fn, lm):
        """Create a Translator from phrase translation table, distance-based
        reordering function and language model."""
        self.table = table
        self.reorder_fn = reorder_fn
        self.lm = lm

    def translate(self, sentence):
        """Translate sentence."""
        max_score = 0
        max_sent = None
        # phrase segmentation
        for i in self.phrase_seg(sentence):
            # reordering
            for j in self.reorder(*i):
                # phrase translation
                for k in self.translate_phrase(*j):
                    # TODO there are possibly many duplicated translation with
                    # different scores here, i.e. (s0, [A, BC, D]) and
                    # (s1, [A, B, C, D]) are identical when they're
                    # concatenated to final translation, so their LM score is
                    # the same. Filtering duplicates and retaining only highest
                    # scored translation before LM-scoring may boost
                    # performance

                    # LM-scoring
                    score, sent = self.lm_score(*k)
                    if score > max_score:
                        max_score = score
                        max_sent = sent
        return max_sent

    def phrase_seg(self, sentence, start=0):
        """Segment sentence into phrases.

        Generates all possible phrase segmentations in which each phrase
        is an entry in phrase translation table. Sentence is given as a list of
        words. Sentence' word index counts from *start*. A phrase segmentation
        is a list of phrases, each includes its start and end index in the
        sentence. Each segmentation is given a score of 1 (i.e. they are
        equally possible). More formally:
        score, phrases := 1.0, [(phrase0, start0, end0), ...]."""
        for i in range(len(sentence)):
            pre = ' '.join(sentence[0:i + 1])
            if pre in self.table:
                # terminate condition: last phrase in sentence
                if not sentence[i + 1:]:
                    yield (1.0, [(pre, start, start + i)])
                # if not terminate, combine this phrase with every possible
                # phrase segmentation of the rest of the sentence
                for score, seg in self.phrase_seg(
                        sentence[i + 1:], start + i + 1):
                    yield (score, [(pre, start, start + i)] + seg)

    def translate_phrase(self, base_score, sentence):
        """Translate using phrase translation table.

        Sentence is given as a list of phrases formatted like phrase_seg's
        output. Generates all possible phrase translations with their
        respective scores multiplying with base score. Each translation is also
        a list of phrases. More formally:
        translation := (score, [phrase0, phrase1,...])."""
        phrase = sentence[0][0]
        # terminate condition: last phrase in sentence
        if len(sentence) == 1:
            for trans_phrase in self.table[phrase]:
                yield (
                    base_score * self.table[phrase][trans_phrase],
                    [trans_phrase]
                )
        else:
            for trans_phrase in self.table[phrase]:
                # if not terminate, combine every possible translation of this
                # phrase with every possible translation of the rest of the
                # sentence
                for score, subsent in self.translate_phrase(
                        base_score, sentence[1:]):
                    yield (
                        self.table[phrase][trans_phrase] * score,
                        [trans_phrase] + subsent
                    )

    def reorder(self, base_score, sentence, end=-1):
        """Reorder all phrases in sentence.

        Sentence is given as a list of phrases formatted like phrase_seg's
        output. End is used for the recursive mechanism of this sentence. It is
        the end index of the phrase right before this subsentence, i.e. use
        default value if not a recursion call. Generates all possible ordering
        with their respective reordering scores multiplying with base score.
        Output format is the same as phrase_seg."""
        # terminate condition: last phrase in sentence
        if len(sentence) == 1:
            yield (
                base_score * self.reorder_fn(sentence[0][1] - end - 1),
                sentence
            )
        for i in range(len(sentence)):
            # if not terminate, combine this phrase with every possible
            # ordering of the rest of the sentence
            for s, j in self.reorder(
                    base_score,
                    sentence[:i] + sentence[i + 1:],
                    sentence[i][2]):
                yield (
                    self.reorder_fn(sentence[i][1] - end - 1) * s,
                    [sentence[i]] + j
                )

    def lm_score(self, base_score, sentence):
        """Calculate LM-score."""
        sentence = ' '.join(sentence).split()
        return (base_score * self.lm.score(sentence), sentence)


class LanguageModelI:
    """Interface for a language model."""
    def __init__(self, sentences):
        pass

    def score(self, sentence):
        """Calculate LM-score for a sentence."""
        raise NotImplementedError()


class TrigramLM(LanguageModelI):
    """Trigram language model."""
    def __init__(self, sentences):
        # FIXME should use smoothing here. I tried SimpleGoodTuringProbDist but
        # it returns zero probability for event with freq=1. Possibly due to
        # too small test corpus
        self.cfd = ConditionalFreqDist(
            (ngram[:-1], ngram[-1])
                for sentence in sentences
                    for ngram in ngrams(sentence, 3, pad_left=True)
        )

    def score(self, sentence):
        score = 1
        for ngram in ngrams(sentence, 3, pad_left=True):
            score *= self.cfd[ngram[:-1]].freq(ngram[-1])
        return score


def d(x):
    """Calculate distance-based reordering cost."""
    return 0.5 ** abs(x)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
