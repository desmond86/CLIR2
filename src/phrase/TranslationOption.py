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
