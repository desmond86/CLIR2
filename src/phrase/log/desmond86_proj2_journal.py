#---------------
#Desmond Putra
#555802
#
#Project 2
#---------------
#
#25 April 2012
#13.00-15.00
#discussing the phrase based model and deocder with the group
#deciding to use EUROPARL corpora. For this project, we only use small
#amount of sentences for training and testing because we focus on the 
#effectiveness program first not the efficiency
#
#
#3 May 2012
#21.00-22.00
#finding information about LM arpa format and how to interpret it.
#
#22.00-24.00
#Code some python program to read language model from SRILM. We use 
#the arpa format that is explained in 
#http://www.speech.sri.com/projects/srilm/manpages/ngram-format.5.html
#
#3 May 2012
#08.00-10.00
#Running moses script to generate LM, reordering model, word alignment. 
#In here, I extracted the sentences that will be used for the training and
#testing. I applied some perl codes that are provided by Moses such as 
#tokenizer and lowercase.
#
#6 May 2012
#12.00-13.00
#discuss with Steven about the reordering model. He gave me some good point 
#about how to count the probability. However, it is a little bit hard to
#implement it.
#
#19 June 2012
#16.00-19.00
#Fixing the Language Model Class. Wrap everything into a standard format 
#for the group. Try setting the github from dmd (which is using proxy)
#
#20 June 2012
#11.00-13.30 & 15.00-19.00
#Discuss with Hai about the progress. Add some functionality in
#phrase extraction algorithm (we use Hai's code from project 1). 
#The functionality that I add:
#1. read english, foreign file and word alignment
#2. format the word alignment to comply with first project code
#3. get probability score for translation model
#4. add another checking/filter to the code
#
#Discuss with Andrew and continue reading the book for more information
#
#21 June 2012
#16.00-22.00
#Add future estimation function. This function follows the pseudo 
#code from Koehn's book but there is an adjustment in line 7 of 
#pseudo code. I tried to check the errata, but it seems there is no
#correction for that one. I already sent an email to Philip just to make
#sure that my understanding is correct. Basically the adjustment that i made
#is about the index position. Maybe in his pseudo code, it is inclusive for 
#the end position. However, we are using python, so we have to add one 
#for the end position (exclusive).
#
#24 June 2012
#21.30-24.00
#Add lexical reordering model to the repo. This lexical reordering check the 
#position of alignment with the previous one. Therea are 3 types of orientation
#such as "monotone", "swap", "discontinue". For the training part, we already
#embedded the code. However, there is a missing link how to use it for
#testing dataset because in the testing dataset there is no information about
#the word alignment
#
#
#25 June 2012
#00.00-01.30
#Continuing the lexical reordering model code
#
#15.00-18.00
#Add another comments to the program. I also upload a number of sentences for 
#testing dataset. 
#
#
#There is another work for installing Moses and all of the stuff (giza, srilm, etc)
#but i did not remember when exactly the time. However, it took me around 2 days
#for installing it on ubuntu :)