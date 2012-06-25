# Hai Dong Luong
# 573780
# Project 2: Phrase-based decoding

"""
First 6 hours on reading, discussion, ideas outlining and presentation.

7 hours on prototyping: pseudo-code as described in Koehn 06 is expanded into
skeleton code in Python with empty functions and classes. Then these functions
and classes are filled in with working code on a top-down order, i.e. top-level
functions are written first, which in the process may require lower-level
functions, which are skeleton at first as well and going to be filled later.

5 hours on Stack & Hypothesis Data structures: added methods to hypothesis data
structure to support sorting in stack (__lt__, __gt__, etc.), printing in
readable format (__str__, __repr__), and stack recombination (identical()).
Stack data structure are filled in with code to add a hypothesis, return the
best hypothesis... When a hypothesis is added, if it is "identical" to an
existing hypothesis in the stack, they are recombined and the worse scored will
be removed.

7 hours on future cost estimation: read future cost estimation code implemented
by other group members and incorporate it into data structures code. Stack's
hypotheses are now sorted by the sum of language, translation and reordering
scores and estimated future cost.

4 hours on stack pruning: histogram and threshold pruning. Pruning is performed
upon every addition. In histogram pruning, when a hypothesis is added and the
size exceeds maximum size, the worst-scored hypothesis is removed. In threshold
pruning, the hypothesis is checked against the best hypothesis and added if it
passes the threshold.

1 hour on reordering limit: the distance between phrases is limited to avoid
excessive searching.

4 hours on modularize and reformat code: initial code was on one big monolithic
file with a few globals. It was modularize to increase reuse and readability.
Code are reformatted in accordance with PEP 8 (using pep8 script
http://pypi.python.org/pypi/pep8)

9 hours on adding docstring and doctest: docstring is added to classes, modules
and functions to clarify their purposes, behaviours and function signatures.
doctest is added to perform basic error checking as well as to act as examples
to users.

3 hours on discussion about performance and shortcomings.
"""
