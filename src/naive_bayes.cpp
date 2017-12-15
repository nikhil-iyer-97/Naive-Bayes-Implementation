#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <utility>

#include "naive_bayes.h"

using namespace std;

NaiveBayesClassifier::NaiveBayesClassifier() {
	;
}

NaiveBayesClassifier::NaiveBayesClassifier(
	int neg_max, int pos_min, const string& train_bow_file,
	const string& vocab_file, const string& sw_file = "") {

	// set some of the private variables
	pos_reviews = 0;
	neg_reviews = 0;
	this -> neg_max = neg_max;
	this -> pos_min = pos_min;
	omit_sw = false;
	if (sw_file != "") {
		stop_words = readWords(sw_file);
		omit_sw = true;
	}
	vocab_words = readWords(vocab_file);
	ll vocab_size = vocab_words.size();

	// Used to store word frequency
	// words_freq[0].first - without binarization
	// words_freq[0].second - with binarization
	//                    .first - freq in positive reviews
	//									  .second -  freq in negative reviews
	vector<pair<pair<ll, ll>, pair<ll, ll>>> words_freq;

	// setup words_freq and words_prob
	words_freq.resize(vocab_size);
	words_prob.resize(vocab_size);
	for (auto& word_info: words_freq) {
		word_info.first.first = 0;
		word_info.first.second = 0;
		word_info.second.first = 0;
		word_info.second.second = 0;
	}

	// populate words_freq
	ifstream in(train_bow_file);
	if (!in.is_open()) {
		cerr << "File opening failed\n";
		exit(0);
	}
	string line;
	ll pos_wobin_freq = 0, neg_wobin_freq = 0, pos_wbin_freq = 0, neg_wbin_freq = 0; // total word frequencies

	// process each bow review in one iteration
	while (getline(in, line)) {

		// obtain sentiment of the review
		stringstream ss;
		ss.str(line);
		ll rating;
		bool is_pos;
		ss >> rating;
		if (rating <= neg_max) {
			is_pos = false;
			++neg_reviews;
		} else if (rating >= pos_min) {
			is_pos = true;
			++pos_reviews;
		} else {
			cerr << "Unexpected Neutral: " << rating << "\n";
			exit(0);
		}

		// process the words encoded as bow and populate words_freq
		ll a, b;
		char discard;
		while (!ss.eof()) {
			ss >> a;
			ss.get(discard);
			ss >> b;
			ss.get(discard);
			if (omit_sw && binary_search(stop_words.begin(), stop_words.end(), vocab_words[a])) {
				continue;
			}
			if (is_pos) {
				words_freq[a].first.first += b;
				pos_wobin_freq += b;
				words_freq[a].second.first += 1;
				++pos_wbin_freq;
			} else {
				words_freq[a].first.second += b;
				neg_wobin_freq += b;
				words_freq[a].second.second += 1;
				++neg_wbin_freq;
			}
		}
	}

	in.close();

	// populate words_prob
	for (ll i = 0; i < vocab_size; i++) {
		words_prob[i].first.first = (1.0 + words_freq[i].first.first) / (pos_wobin_freq + vocab_size);
		words_prob[i].first.second = (1.0 + words_freq[i].first.second) / (neg_wobin_freq + vocab_size);
		words_prob[i].second.first = (1.0 + words_freq[i].second.first) / (pos_wbin_freq + vocab_size);
		words_prob[i].second.second = (1.0 + words_freq[i].second.second) / (neg_wbin_freq + vocab_size);
	}
}

void NaiveBayesClassifier::test(const string& test_bow_file, bool use_bin) {
	ifstream in(test_bow_file);
	if (!in.is_open()) {
		cerr << "File opening failed\n";
		exit(0);
	}

	ll tp = 0, fp = 0, fn = 0, tn = 0;
	string line;
	// consider one review in each iteration
	while (getline(in, line)) {
		// obtain the sentiment of the review
		ll rating;
		stringstream ss;
		ss.str(line);
		ss >> rating;
		bool is_pos;
		if (rating <= neg_max) {
			is_pos = false;
		} else if (rating >= pos_min) {
			is_pos = true;
		} else {
			cerr << "Unexpected Neutral: " << rating << "\n";
			exit(0);
		}

		// classify the instance
		if (classify(ss, use_bin)) {
			if (is_pos) {
				++tp;
			} else {
				++fp;
			}
		} else {
			if (is_pos) {
				++fn;
			} else {
				++tn;
			}
		}
	}
	in.close();

	// print statistics
	cout << fixed << setprecision(6)
	<< "Accuracy: " << (static_cast<double>(tp + tn) / (tp + tn + fp + fn)) * 100 << "%\n"
	<< "Precision: " << (static_cast<double>(tp)) / (tp + fp) << "\n"
	<< "Recall: " << (static_cast<double>(tp)) / (tp + fn) << "\n"
	<< "F1 Measure: " << (2 * static_cast<double>(tp)) / (2 * tp + fp + fn) << "\n";
}

vector<string> NaiveBayesClassifier::mostInformative(ll num, bool use_bin) {
	// find P(w/+ve sentiment) / P(w/-ve sentiment) for each word
	vector<pair<ld,ll>> temp;
	for(int i =0 ;i<words_prob.size(); i++) {
		if(!use_bin)
			temp.push_back(make_pair(words_prob[i].first.first/words_prob[i].first.second,i));
		else temp.push_back(make_pair(words_prob[i].second.first/words_prob[i].second.second,i));
	}
	sort(temp.begin(),temp.end());
	reverse(temp.begin(),temp.end());

	// return num number of features
	vector<string> return_vec;
	for(int i=0; i<num; i++)	return_vec.push_back(vocab_words[temp[i].second]);
	return return_vec;
}

vector<string> NaiveBayesClassifier::readWords(const string& sw_file) {
	ifstream fin(sw_file,ios::in);
	vector<string> data;

	while(!fin.eof()){
		string s;
		fin>>s;
		stringstream str(s);
		data.push_back(s);
	}
	return data;
}

bool NaiveBayesClassifier::classify(stringstream& bow_review_instance, bool use_bin) {
	stringstream& ss = bow_review_instance;
	ld pos_prob = log(static_cast<ld>(pos_reviews) / (pos_reviews + neg_reviews));
	ld neg_prob = log(static_cast<ld>(neg_reviews) / (pos_reviews + neg_reviews));

	ll a, b;
	char discard;
	while (!ss.eof()) {
		ss >> a;
		ss.get(discard);
		ss >> b;
		ss.get(discard);

		// skip if this is a stopword
		if (omit_sw && binary_search(stop_words.begin(), stop_words.end(), vocab_words[a])) {
			continue;
		}
		if (use_bin) {
			pos_prob += log(words_prob[a].second.first);
			neg_prob += log(words_prob[a].second.second);
		} else {
			pos_prob += (b * log(words_prob[a].first.first));
			neg_prob += (b * log(words_prob[a].first.second));
		}
	}
	return (pos_prob >= neg_prob ? true : false);
}
