#ifndef NAIVE_BAYES_H_
#define NAIVE_BAYES_H_

#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <fstream>
#include <map>

#define ll long long
#define ld long double

using namespace std;

class NaiveBayesClassifier {
public:
	// Default constructor
	NaiveBayesClassifier();

	// Using this constructor would train the Naive Bayes classifer
	// Parameters:
	//   neg_max - Maximum rating to be considered as a negative sentiment
	//   pos_min - Minimum rating to be considered as a positive sentiment
	//   train_bow_file - Path to the file which contains the training data in
	//                    Bag of Words format
	//   vocab_file - Path to the file which contains the vocabulary
	//   sw_file - Path to the file which contains the stopwords
	NaiveBayesClassifier(
		int neg_max, int pos_min, const string& train_bow_file,
		const string& vocab_file, const string& sw_file);

	// Tests the trained Naive Bayes classifier and prints statistics
	// Parameters:
	//   test_bow_file - Path to the file which contains the testing data in
	//                   Bag of Words format
	//   use_bin - true if binarization is to be used
	void test(const string& test_bow_file, bool use_bin);

	// Returns the most informative features
	// Parameters:
	// 	num - Number of features to return
	//  use_bin - true if binarization is to be considered
	vector<string> mostInformative(ll num, bool use_bin);

private:
	// Returns a vector of words read from a file
	// Parameters:
	//   sw_file - Path to the file which contains one word on each line
	vector<string> readWords(const string& sw_file);

	// Returns the classification made for an instance
	// Parameters:
	//   bow_review_instance - A stringstream which contains the review
	//                         (in Bag of Words format) to be classified
	//   use_bin - true if binarization is to be used
	bool classify(stringstream& bow_review_instance, bool use_bin);

	// Conditional probability of occurence of words given sentiment of review
	// words_prob[i].first - without binarization
	// words_prob[i].second - with binarization
	//                    .first - prob given positive
	//									  .second -  prob given negative
	vector<pair<pair<ld, ld>, pair<ld, ld>>> words_prob;

	// Stores the stop words. Populated by the non-default constructor
	vector<string> stop_words;

	// Stores the words in the vocabulary. Populated by the non-default
	// constructor
	vector<string> vocab_words;

	// Total number of positive reviews
	ll pos_reviews;

	// Total number of negative reviews
	ll neg_reviews;

	// Maximum rating to be considered as a negative sentiment
	int neg_max;

	// Minimum rating to be considered as a positive sentiment
	int pos_min;

	// true if stopwords should be omitted
	bool omit_sw;
};

#endif
