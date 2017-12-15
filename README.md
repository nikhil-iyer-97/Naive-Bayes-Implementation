# Naive-Bayes-Implementation

The dataset is a collection of movie reviews and their corresponding sentiment values(<=4 for negative and >=7 for positive).
The dataset also contains unclassified reviews and the task is to train the Naive-Bayes classifier to correctly predict the ratings for the unsupervised text.

The .feat files in data folder includes tokenized bag of words features which is the primary source of training and testing data for the classifier. The imdb.vocab file is used to obtain the word corresponding to a given index whenever needed.

The src directory contains the source code for the classifier and takes its input from the .feat files to predict ratings for the movies not rated and check for precision and accuracy.

Usage:
------
-open src directory and run "g++ *.cpp" to run all the files simultaneously.
