from ModelExtractor import *

#language model
lm = SRILangModel()

#read language model file
lm.read_lm_file("../source_files/all.lm")

#find the score (log10)
output_lm = lm.get_language_model_prob("accommodated")
print output_lm

#translation model
english_file = "../source_files/all.lowercased.raw.en"
foreign_file = "../source_files/all.lowercased.raw.fr"
alignment_file = "../source_files/aligned.grow-diag-final-and"

#run the translation model
tm = TranslationModel(english_file, foreign_file, alignment_file)
tm.extract()

#find list of translations
output_tm = tm.get_translation_model_prob("en")

#find the score (log10) sort by highest score
for key, value in sorted(output_tm.iteritems(), key=lambda (k,v): (v,k), reverse=True):
    print key, value
