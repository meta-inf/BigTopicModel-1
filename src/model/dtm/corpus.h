//
// Created by dc on 7/4/16.
//

#ifndef DTM_CORPUS_H
#define DTM_CORPUS_H

#include <Eigen>
#include <vector>
#include <string>

using std::vector;
using std::pair;
using std::string;
using Eigen::ArrayXXd;
using Eigen::ArrayXd;

struct EpCorpus {
	ArrayXXd words; // (K_doc, K_vocab)
	struct Token {
		int w, f;
	};
	vector<vector<Token>> tokens;
	int K_doc;
	double K_tokens;
	EpCorpus(const ArrayXXd &m);
};

struct Corpus {
	int K_vocab;
	vector<EpCorpus> epochs;
	vector<string> vocab;
	Corpus () {}
	Corpus (const string &corpus_path, const string &dict_path);
	static Corpus Merged (const Corpus &c);
};


#endif //DTM_CORPUS_H
