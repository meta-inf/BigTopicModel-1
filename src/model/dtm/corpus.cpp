//
// Created by dc on 7/4/16.
//

#include "corpus.h"
#include <fstream>
using namespace std;

EpCorpus::EpCorpus (const ArrayXXd &m): words(m), K_doc(m.rows()), K_tokens(0) {
}

Corpus::Corpus (const string &corpus_path, const string &dict_path) {
	ios::sync_with_stdio(0);

	// init corpus
	fstream fcorp(corpus_path);
	fcorp >> K_vocab;
	while (1) {
		string h;
		fcorp >> h;
		assert(h == "YR");
		int yr, K_doc;
		fcorp >> yr;
		if (yr == -1) {
			break;
		}
		fcorp >> K_doc;
		EpCorpus ts{ArrayXXd::Zero(K_doc, K_vocab)};
		ts.tokens.resize((size_t)K_doc);
		ts.K_tokens = 0;
		for (int i_doc = 0; i_doc < K_doc; ++i_doc) {
			int nz, w, freq;
			for (fcorp >> nz; nz--; ) {
				fcorp >> w >> freq;
				ts.words(i_doc, w) = freq;
				ts.tokens[i_doc].push_back(EpCorpus::Token{w, freq});
				ts.K_tokens += freq;
			}
		}
		epochs.push_back(ts);
	}

	// Init dictionarty
	fstream fdict(dict_path);
	vocab.resize(K_vocab);
	for (int d = 0; d < K_vocab; ++d) {
		string w;
		int i;
		fdict >> w >> i;
		vocab[i] = w;
	}
}

Corpus Corpus::Merged (const Corpus &c) {
	Corpus ret;
	ret.vocab = c.vocab;
	ret.K_vocab = c.K_vocab;
	int K_doc = 0;
	for (auto &e: c.epochs) {
		K_doc += e.K_doc;
	}
	EpCorpus ec(ArrayXXd::Zero(K_doc, c.K_vocab));
	int cur_row = 0;
	for (auto &ecs: c.epochs) {
		for (int d = 0; d < ecs.K_doc; ++d) {
			ec.words.row(cur_row++) = ecs.words.row(d);
		}
		ec.tokens.insert(ec.tokens.end(), ecs.tokens.begin(), ecs.tokens.end());
		ec.K_tokens += ecs.K_tokens;
	}
	ret.epochs.push_back(ec);
	return ret;
}
