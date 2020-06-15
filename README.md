# NLP and Pipelines
 
Natural language processing is one of the fastest growing fields in the world.This Repo begins with an overview of how to design an end-to-end NLP pipeline where you start with raw text, in whatever form it is available, process it, extract relevant features, and build models to accomplish various NLP tasks. 

NLP Pipelines consist of tree stages:

<p align="center">
  <img src="/imgs/1.PNG" alt="" width="500" height="150" >
 </p>
 
   * **Text Processing**: Take raw input text, clean it, normalize it, and convert it into a form that is suitable for feature extraction.
   * **Feature Extraction**: Extract and produce feature representations that are appropriate for the type of NLP task you are trying to accomplish and the type of model you are planning to use.

   * **Modeling**: Design a statistical or machine learning model, fit its parameters to training data, use an optimization procedure, and then use it to make predictions about unseen data.
   
   
**This process isn't always linear and may require additional steps to go back and forth.**

## Stage 1: Text Processing
In this step we will explore the steps involved in **text processing**, but before, the aquestion of why we need to Process Text or why we can not feed directly, should be answered.


 * **Extracting plain text**: Textual data can come from a wide variety of sources: the web, PDFs, word documents, speech recognition systems, book scans, etc. Your goal is to extract plain text that is free of any source specific markup or constructs that are not relevant to your task.
 
 * **Reducing complexity**: Some features of our language like capitalization, punctuation, and common words such as a, of, and the, often help provide structure, but don't add much meaning. Sometimes it's best to remove them if that helps reduce the complexity of the procedures you want to apply later.


**The following presented the text processing steps to prepare data from different sources:**

 1. **Cleaning** to remove irrelevant items, such as HTML tags
 
 2. **Normalizing** by converting to all lowercase and removing punctuation
 
 3. Splitting text into words or **tokens**
 
 4. Removing words that are too common, also known as **stop words**
 
 5. Identifying different **parts of speech** and **named entities**
 
 6. Converting words into their dictionary forms, using **stemming and lemmatization**

**After performing these steps above, your text will capture the essence of what was being conveyed in a form that is easier to work with.**


### 1. Cleaning: 

In this section I am going to walk through the step of cleaning text data from a popular source - **the web**. You'll be introduced to helpful tools in working with this data, including the [requests library](http://docs.python-requests.org/en/master/user/quickstart/#make-a-request), [regular expressions](https://docs.python.org/3/library/re.html), and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/).

Check out this [jupyter notebook](https://github.com/A2Amir/NLP-and-Pipelines/blob/master/codes/1_cleaning.ipynb) for more details about Cleaning. 

### 2. Normalization: 

Plain text is great but it's still human language with all its variations. we'll try to reduce some of that complexity.  **Lowercase conversion** and **punctuation removal** are the two most common text normalization steps.  

*  In the English language, the starting letter of the first word in any sentence is usually capitalized.  While this is convenient for
a human reader, from the standpoint of a machine learning algorithm, it does not make sense to differentiate between them. (for example Car,car, and CAR), they all mean the same thing. Therefore, we usually convert every letter in our text to a common case,usually lowercase, so that each word is represented by a unique token. 

*  you may want to remove special characters like periods, question marks, and exclamation points from the text and only keep letters of the alphabet and maybe numbers. This is especially useful when we are looking at text documents as a whole in applications like document classification and clustering where the low level details do not matter a lot. 

* Is it better to just replace punctuation characters with a space, because words don't get concatenated together, in case the original text did not have a space before or after the punctuation.

Check out this [jupyter notebook](https://github.com/A2Amir/NLP-and-Pipelines/blob/master/codes/2_normalization.ipynb) for more details about Normalization. 


### 3. Tokenization: 

Token is a fancy term for a symbol. Usually, one that holds some meaning and is not typically split up any further. In case of natural language processing, our tokens are usually **individual words**. So tokenization is simply splitting each sentence into a sequence of words. Check out this [jupyter notebook](https://github.com/A2Amir/NLP-and-Pipelines/blob/master/codes/3_tokenization.ipynb) and the [nltk.tokenize package](http://www.nltk.org/api/nltk.tokenize.html) for more details. 


### 4. Stop Word Removal: 

Stop words are uninformative words (like, is, our, the, in, at, etc...) that do not add a lot of meaning to a sentence.
They are typically very commonly occurring words, and we may want to remove them to reduce the vocabulary we have to
deal with and hence the complexity of later procedures. 

* Note that stop words are based on a specific corpus and different corpora may have different stop words. 
* A word maybe a stop word in one application, but a useful word in another. 

Check out this [jupyter notebook](https://github.com/A2Amir/NLP-and-Pipelines/blob/master/codes/4_stop_words.ipynb) for more details about Stop Word Removal. 

### 5. Part-of-Speech Tagging:

Identifying how words are being used in a sentence as Nouns, pronouns, verbs, adverbs, etc..., can help us better understand what is being said. It can also point out relationships between words and recognize cross references. 

<p align="center">
  <img src="/imgs/2.PNG" alt="" width="500" height="200" >
 </p>

**Note:** Part-of-speech tagging using [the pos_tag function](https://www.nltk.org/book/ch05.html)  in the nltk library can be very tedious and error-prone for a large corpus of text, since you have to account for all possible sentence structures and tags!

There are other more advanced forms of POS tagging that can learn sentence structures and tags from given data, including Hidden Markov Models (HMMs) and Recurrent Neural Networks (RNNs).


### 6. Named Entity Recognition: 
 
Named entities are typically noun phrases that refer to some specific object, person, or place.  Named entity recognition is often used to index and search for news articles, for example, on companies of interest. Check out this [jupyter notebook](https://github.com/A2Amir/NLP-and-Pipelines/blob/master/codes/5_pos_ner.ipynb) for more details about Part-of-Speech Tagging and Named Entity Recognition. 

### 7. Stemming and Lemmatization:

Stemming is the process of reducing a word to its stem or root form. For instance, branching, branched, branches, can all be reduced to branch. This helps reduce complexity while retaining the essence of meaning that is carried by words.  Stemming is meant to be a fast and crude operation carried out by applying very simple search and replace style rules.  This may result in stem words that are not complete words, but that's okay, as long as all forms of that word are reduced to the same stem. 



Lemmatization is another technique used to reduce words to a normalized form., The transformation of lemmatization actually uses a dictionary to map different variants of a word back to its root.  With this approach, we are able to reduce non-trivial inflections such as is, was, were, back to the root 'be'.


#### Differences

 * Stemming sometimes results in stems that are not complete words in English. Lemmatization is similar to stemming with one difference, the final form is also a meaningful word.
 
* Stemming does not need a dictionary like lemmatization does.

* Depending on the constraints you have, stemming maybe a less memory intensive option for you to consider. 


Check out this [jupyter notebook](https://github.com/A2Amir/NLP-and-Pipelines/blob/master/codes/6_stem_lemmatize.ipynb) for more details about Stemming and Lemmatization. 

### summarize what a typical workflow looks like: 

1. Starting with a plain text sentence, 
2. Normalize it by converting to lowercase and removing punctuation, 
3. Split it up into words using a tokenizer. 
4. Remove stop words to reduce the vocabulary you have to deal with. 
5. Depending on your application, you may then choose to apply a combination of stemming and lemmatization to reduce words to the root or stem form. It is common to apply both, lemmatization first, and then stemming. 

## Stage 2: Feature Extraction
Words comprise of charachters which are just sequences of ASCII or Unicode values and computers don't have a standard representation for words. Computer don't quite capture the meanings or relationships between words. 
<p align="center">
  <img src="/imgs/3.PNG" alt="" width="400" height="300" >
 </p>

The question is,how do we come up with a representation for text data that we can use as features for modeling?  The answer depends on what kind of model you're using and what task you're trying to accomplish. 


* If you want to use a graph based model to extract insights, you may want to represent your words as symbolic nodes with relationships between them like WordNet. 


* If you're trying to perform a document level task, such as spam detection or sentiment analysis,you may want to use a per document representations such as bag-of-words or doc2vec. 

* If you want to work with individual words and phrases such as for text generation or machine translation, you'll need a word level representation such as word2vec or glove. 
<p align="center">
  <img src="/imgs/4.PNG" alt="" width="600" height="400" >
 </p>


* There are many ways of representing textual information and it is only through practice that you can learn what you need for each problem. 

### 1. Bag of Words:

The Bag of Words model treats each document as an un-ordered collection of words. Here, a document is the unit of text that you want to analyze. For instance each essay or tweet, would be a document. To obtain a bag of words from a piece of raw text,you need to apply appropriate text processing steps (explained above) then treat the resulting tokens as an un-ordered collection.

<p align="center">
  <img src="/imgs/5.PNG" alt="" width="400" height="300" >
 </p>



After producing set of words from documents, keeping these as separate sets  is very inefficient. They're of different sizes, contain different words,  hard to compare and words occur multiple times in each document.

A better representation is creating a set of documents , which is known as a corpus and turning each document of the corpus into a vector of numbers representing how many times each unique word in the corpus occurs in a document.  

<p align="center">
  <img src="/imgs/6.PNG" alt="" width="500" height="400" >
 </p>
 
what you can do with this representation is to compare two documents based on how many words they have in common or how similar their term frequencies are:

* A more mathematical way of expressing that is to compute the dot product between the two row vectors, which is the sum of the products of corresponding elements. Greater the dot product,more similar the two vectors are.  


<p align="center">
  <img src="/imgs/7.PNG" alt="" width="400" height="200" >
 </p>
The dot product has one flaw, it only captures the portions of overlap.It is not affected by other values that are not uncommon. 

* A better measure is cosine similarity,where we divide the dot product of two vectors by the product of their magnitudes or Euclidean norms. 

<p align="center">
  <img src="/imgs/8.PNG" alt="" width="400" height="300" >
 </p>

 If you think of these vectors as arrows in some n-dimensional space, then this is equal to the cosine of the angle theta between them.
Identical vectors have cosine equals one. Orthogonal vectors have cosine equal zero and for vectors that are exactly opposite, it is minus one. So, the values always range nicely between one for most similar, to minus one, most dissimilar.
