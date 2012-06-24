#Andrew Vadnal
#326558
#[date]
#Project 2 (Phrase Based Model Extension)
#Journal

''' 
============== The allocated time for this project is 48 hours. ==============

Hour 1-2. Begin reading Koehn chapter 6. -
http://langtech.github.com/clir/materials/koehn-06.pdf
Forked clir repository, shared it with team members. Uploaded team members
project 1 files as a reference for all.

Hour 3. Spent in tutorial. Discussed with team about the requirements for
next Wednesday's presentation. These requirements include: Setting up a work
plan, establishing project milestones, how we are going to utilise existing
resources (such as using tools to generate the required models:
translation/langauge/reordering) setting up team member responsibilities and
ensuring that we can present a progress report.

Hour 4. Brainstorming an appopriate data structure that can be used for
storing 'hypotheses' as mentioned in the aforementioned Koehn paper. Continued
reading Koehn paper up until section 6.4. Will revisit this when implementing
optimisations. 

Hour 5-6. Spent in the workshop. Programmatically constructing the data
structure for the hypothesis with Hai. An Hypothesis will be a class, allowing
the attributes: translation, partial score and a pointer to the previous
hypothesis. Have to work on processing the phrase table output generated from
project 1.

Hour 7-10. Been a while since I've worked on this project due to exams, just
reading through notes and refreshing my mind on key topics, reading through
the phrase paper once again, and re-understanding the code from the previous
project.

Hour 11-16. Met up with team. Figuring out why team members can't clone my github repo. Implementing the pseudo-code with Hai in
the koehn-06 Decoding chapter pg 165, 170. Setting up an overall structure "decoder.py", so each member can contribute to filling in the 
scaffolding - ie. the functions which have not yet been implemented. Spent time working on histogram and threshold pruning
in addition to the section on determing future cost for any input span. Did not have access to model data (translation/language)
while doing this, which was to be committed at the end of the day.

Hour 17-19. Merging code (both team members work from uni - proxy doesn't allow
push/pull/commits to github), running tests with LM/TM code in separate file.

Hour 20-22. Determining how to incorporate the LM/TM code into the pruning process based on tests performed
on the separate file. Begin substituting dummy data with the real model data in test file.

Hour 23-29. Figuring out how to extract the correct information for a given sentence. I wasn't sure if what I had
was working or not, so as a means of testing I used the dummy data from koehn-06 to reconstruct and replicate
their experiment. If this was successful, I would know my approach was correct. Seems to be working, ie. replicating
the future cost table as per specified in the koehn-06 errata. Next step is to merge all test code into a more concrete
implementation, that is - using all obtained costs from the model. (FC/TM/LM)

Hour 30-37. Finished implementation of future cost estimation based on a given input phrase. TM/LM costs are also
being correctly extracted from the models. Adding comments to functions and also deleting redundant code. Next step
is to use all these phrasal costs in the threshold pruning process.

'''
