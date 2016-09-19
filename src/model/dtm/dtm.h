//
// Created by dc on 7/4/16.
// TODO: thread safety of random engines
//

#ifndef DTM_DTM_H
#define DTM_DTM_H


#include "corpus.h"
#include "aliastable.h"
#include "random.h"

using Eigen::ArrayXXd;
using Eigen::ArrayXXi;


struct EpochSample {
	ArrayXXd alpha; // (1, K_topic-1)
	ArrayXXd eta;   // (K_doc, K_topic-1)
	ArrayXXd phi;   // (K_topic, K_vocab-1)
	ArrayXXd phiAux; // phi.shape, auxiliary in pSGLD / SGHMC
	int t_sgld_eta, t_sgld_phi;

	// Count tables for Z
	ArrayXXd cdt;   // (K_doc_batch, K_topic)
	ArrayXXd nd;    // (K_doc_batch, 1)
	ArrayXXd cwt;	// (K_topic, K_vocab)
	ArrayXXd ct;    // (K_topic, 1)
	// End count tables for Z

	struct Batch {
		vector<int> doc_id;
		size_t size () const { return doc_id.size(); }
	} b;

	const EpCorpus *corpus;
};

class DTM {
public:

	const vector<string> vocab;
	vector<EpochSample> sample_[2]; // Test_Full
	vector<EpCorpus> test_held_, test_observed_;
	const int K_topic, K_vocab, K_doc_batch, N_gibbs_step;
	const double report_every_, dump_every_;
	int cur_sample_idx_;

	std::random_device rd_device_;

	// th - test_held, to - test_observed
	DTM (const Corpus &corpus, const Corpus &corpus_th, const Corpus &corpus_to, int N_gibbs_step,
		 double report_every, double dump_every, EpochSample *single_ctm);

	struct UpdateWorker { // For multi-thread
		rand_data *rd_data_;
		const int K_topic, K_vocab, N_gibbs_step;
		int K_doc_batch;
		// FIXME: Debugging info
		vector<double> _p_act;
		double _sum_p_act;
		vector<double> _q_act;
		double _sum_q_act;
		vector<AliasTable> alt_word_, alt_doc_;
		UpdateWorker (DTM &par, int seed);
		~UpdateWorker ();
		void InitEpochSample (EpochSample *s, const ArrayXXd &phi, const ArrayXXd &alpha, const EpCorpus *corpus);
		void SampleBatch (EpochSample *st);
		void BuildDocAlias (const EpochSample *s);
		void UpdateZ (EpochSample *st, const EpochSample *st0);
		void UpdateAlpha (EpochSample *st, const EpochSample *st0, const EpochSample *stm1, const EpochSample *stp1);
		void UpdateEta (EpochSample *st, const EpochSample *st0);
		void UpdatePhi (EpochSample *st, const EpochSample *st0, const EpochSample *stm1, const EpochSample *stp1);
		double EstimateLL (const EpochSample *train, const EpCorpus &test_observed, const EpCorpus &test_held);
	};
	vector<UpdateWorker> workers_;
	void ShowTopics (const EpochSample *st, int K);
	void DumpSample (std::string path);
	void Infer ();
	void DebugPhi ();
};

// Some previously used hyperparameters
//	static constexpr double stddev_al0 = 0.1, stddev_phi0 = 10, stddev_eta = 4;
//	static constexpr double stddev_al0 = 0.1, stddev_phi0 = 6, stddev_eta = 3;
//	static constexpr double stddev_al = 0.6, stddev_phi = 0.2;

#endif //DTM_DTM_H
