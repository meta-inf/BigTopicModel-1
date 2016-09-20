#ifndef LCORPUS_H
#define LCORPUS_H

#include "utils.h"

struct Token {
	int w, f;
};

struct LocalCorpus {
	int ep_s, ep_e; // [s, e), same below
	struct Doc {
		vector<Token> tokens;
	};
	vector<vector<Doc>> docs;
	int vocab_s, vocab_e; 

    LocalCorpus(const std::string &fileName);
};

#endif
