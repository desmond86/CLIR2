"""
Provides utilitiy functions for decoding process.

Authors:
Hai Dong Luong (573780) <hai-ld>
Desmond Putra () <dputra>
Andrew Vadnal (326558) <avadnal>
"""
# custom modules
from TranslationOption import TranslationOption


def get_all_phrases(sentence):
    """Get all phrases in a sentence.
    sentence: a sentence given as a list of words

    >>> list(get_all_phrases('Stairway to heaven'.split())) \
            # doctest: +NORMALIZE_WHITESPACE
    [['Stairway'], ['Stairway', 'to'], ['Stairway', 'to', 'heaven'], ['to'], \
    ['to', 'heaven'], ['heaven']]
    """
    for i in range(len(sentence)):
        for j in range(i + 1, len(sentence) + 1):
            yield sentence[i:j]


def get_trans_opts(hyp, tm, rm, lm):
    """Get all translation options a hypothesis could be expanded upon.

    hyp: the hypothesis to be expanded
    
    tm, rm, lm: TranslationModel, ReorderingModel and SRILangModel,
    respectively

    Return a generator of TranslationOption for hyp. Each TranslationOption is
    weighted using the three models.

    >>> import random
    >>> trans_opts = list(get_trans_opts(empty_hypothesis, tm, rm, lm))
    >>> random.seed(314159)
    >>> random.shuffle(trans_opts)
    >>> [(t.input_phrase, t.output_phrase) for t in trans_opts] \
            # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    [(['le'], 'the present'), \
        (['le'], 'on the other'), \
        (['pour', 'le'], 'for the'),...
    """
    untrans = get_untranslated_words(hyp)

    for part in get_consecutive_parts(untrans):
        for phrase in get_all_phrases(part):
            try:
                reordering_score = rm.score(hyp.trans['input'][-1], phrase)
            except IndexError:
                reordering_score = 0.0
            if reordering_score is None:
                continue
            for translation in get_translations(phrase, tm):
                try:
                    words = hyp.trans['output'][-1].split() + \
                        translation.output_phrase.split()
                except IndexError:
                    # empty translation output => start of output sentence
                    words = ['<s>'] + translation.output_phrase.split()

                ngrams = [
                    (words[i], words[i + 1]) for i in range(len(words) - 1)
                ]
                language_score = sum([
                    lm.get_language_model_prob(' '.join(ngram))
                        for ngram in ngrams
                ])
                translation.score += reordering_score + language_score
                yield translation


def get_untranslated_words(hyp):
    """Get words untranslated by a hypothesis.

    Returns a list of tuples in which second elements are untranslated words
    and first elements are positions of the words in input sentence.

    >>> get_untranslated_words(empty_hypothesis)
    [(0, 'pour'), (1, 'le'), (2, 'moment')]
    >>> trans_opt = TranslationOption(1, 1, ['le'], 'the', 0.0)
    >>> hypothesis = Hypothesis(empty_hypothesis, trans_opt)
    >>> get_untranslated_words(hypothesis)
    [(0, 'pour'), (2, 'moment')]
    """
    input_sent = dict(enumerate(hyp.input_sent))
    for i in hyp.trans['input']:
        for j in range(i[0], i[1] + 1):
            del input_sent[j]

    return input_sent.items()


def get_consecutive_parts(input_sent):
    """Get consecutive parts in a non-consecutive sentence.

    input_sent: input sentence given as a list of tuples in which second
    elements are untranslated words and first elements are positions of the
    words in input sentence.

    Return a list of consecutive parts, each of which is a list of words.

    >>> get_consecutive_parts([(1, 'a'), (2,  'b'), (3, 'c'),
    ... (5, 'e'), (6, 'f'), (8, 'h')])
    [[(1, 'a'), (2, 'b'), (3, 'c')], [(5, 'e'), (6, 'f')], [(8, 'h')]]
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


def get_translations(phrase, tm):
    """Generate translations of a phrase using the specified translation model.

    phrase: a phrase given as a list of tuples in which second elements are
    untranslated words and first elements are positions of the words in input
    sentence, e.g [(1, 'a'), (2,  'b'), (3, 'c'),]

    tm: a translation model

    Return generator for phrase translation, elements are of TranslationOption

    >>> [t.output_phrase for t in get_translations(
    ... [(0, 'pour'), (1, 'le'), (2, 'moment')], tm)]
    ['at present']
    """
    phrase_words = [p[1] for p in phrase]
    translations = tm.get_translation_model_prob_f(' '.join(phrase_words))
    for translation, score in translations.iteritems():
        t = TranslationOption(phrase[0][0], phrase[-1][0],
            phrase_words, translation, score)
        yield t

if __name__ == '__main__':
    import doctest
    from ModelExtractor import SRILangModel, TranslationModel
    from ReorderingModel import ReorderingModel
    from datastructures import Hypothesis
    from collections import defaultdict

    lm = SRILangModel()
    rm = ReorderingModel()

    lm.read_lm_file("source_files/all.lm")

    english_file = "source_files/all.lowercased.raw.en"
    foreign_file = "source_files/all.lowercased.raw.fr"
    alignment_file = "source_files/aligned.grow-diag-final-and"

    tm = TranslationModel(english_file, foreign_file, alignment_file)
    tm.extract()

    fc_table = defaultdict(lambda: defaultdict(float))
    empty_hypothesis = Hypothesis(None, None, 'pour le moment'.split(), fc_table)

    _globals = {
        'lm': lm,
        'rm': rm,
        'tm': tm,
        'empty_hypothesis': empty_hypothesis,
    }
    doctest.testmod(extraglobs=_globals)
