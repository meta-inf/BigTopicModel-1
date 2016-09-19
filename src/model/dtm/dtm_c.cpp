//
// Created by dc on 7/4/16.
//

#include <cmath>
#include <cstdlib>
#include <gflags/gflags.h>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include "utils.h"
#include "dtm.h"

using namespace std;

DEFINE_bool(fix_random_seed, true, "Fix random seed for debugging");
DEFINE_bool(show_topics, false, "Display top 10 words in each topic");
DEFINE_int32(n_sgld, 2, "number of sgld iterations for phi");
DEFINE_int32(n_sgld_e, 2, "number of sgld iterations for eta");
DEFINE_int32(n_mh_steps, 16, "number of burn-in mh iterations for Z");
DEFINE_int32(n_infer_burn_in, 16, "number of burn-in steps in test");
DEFINE_int32(n_infer_samples, 1, "number of samples used in test");
DEFINE_int32(n_threads, 2, "number of threads used");
DEFINE_int32(n_topics, 50, "number of topics");
DEFINE_int32(n_doc_batch, 60, "implemented");
DEFINE_bool(sgld_mh, false, "Do MH test in SGLD");
DEFINE_bool(fix_alpha, false, "Use fixed symmetrical topic prior");
DEFINE_double(sgld_phi_a, 0.5, "");
DEFINE_double(sgld_phi_b, 100, "");
DEFINE_double(sgld_phi_c, 0.8, "");
DEFINE_double(sgld_eta_a, 0.5, "");
DEFINE_double(sgld_eta_b, 100, "");
DEFINE_double(sgld_eta_c, 0.8, "");
DECLARE_string(dump_prefix);

// dst(i, j) ~ N(mean(i, j), var)
void SampleNormal (ArrayXXd *dst, const ArrayXXd &mean, double var, rand_data *rd) {
	double stddev = sqrt(var);
	NormalDistribution normal_rnd_;
	*dst = ArrayXXd(mean.rows(), mean.cols());
	for (int i = 0; i < mean.rows(); ++i) {
		for (int j = 0; j < mean.cols(); ++j) {
			(*dst)(i, j) = normal_rnd_(rd) * stddev + mean(i, j);
		}
	}
}

// dst(i, j) ~ N(mean(0, j), var)
void SampleNormalBatch (ArrayXXd *dst, const ArrayXXd &mean, double var, int size, rand_data *rd) {
	double stddev = sqrt(var);
	NormalDistribution normal_rnd_;
	assert(mean.rows() == 1);
	*dst = ArrayXXd(size, mean.cols());
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < mean.cols(); ++j) {
			(*dst)(i, j) = normal_rnd_(rd) * stddev + mean(0, j);
		}
	}
}

// dst(i, j) = exp(src(i, j)) / sum_j'(exp(src(i, j'))
inline ArrayXXd RowwSoftmax (const ArrayXXd &src) {
	// Row-major; R^N => R^N
	ArrayXXd dst = src;
	dst.colwise() -= dst.rowwise().maxCoeff();
	dst = Eigen::exp(dst);
	dst.colwise() /= dst.rowwise().sum();
	return dst;
}

inline ArrayXXd Add0 (const ArrayXXd &src) { // Reduced-natural -> expaneded-natural
	return src;
}
inline ArrayXXd Del0 (const ArrayXXd &src) { // e->R
	return src;
}
inline ArrayXd logsumexp (const ArrayXXd &src) {
	ArrayXXd log_src = src;
	ArrayXd maxC = log_src.rowwise().maxCoeff();
	log_src.colwise() -= maxC;
	return Eigen::log(Eigen::exp(log_src).rowwise().sum()) + maxC;
}
inline ArrayXXd LogRowwSoftmax (const ArrayXXd &src) { // Log(RowwSoftmax(src)). save some logs
	ArrayXXd log_src = src;
	log_src.colwise() -= logsumexp(log_src);
	return log_src;
}

// Allocate and init a sample
void DTM::UpdateWorker::InitEpochSample (EpochSample *s, const ArrayXXd &phi, const ArrayXXd &alpha, const EpCorpus *corpus) {
	s->corpus = corpus;
	s->alpha = alpha;
	SampleNormalBatch(&s->eta, s->alpha, 0, s->corpus->K_doc, rd_data_);
	s->phi = phi;
	s->t_sgld_eta = s->t_sgld_phi = 0;
}

DTM::UpdateWorker::UpdateWorker (DTM &p, int seed) :
	K_topic(p.K_topic), K_vocab(p.K_vocab), K_doc_batch(p.K_doc_batch), N_gibbs_step(p.N_gibbs_step),
	_sum_p_act(0)
{
	rd_data_ = new rand_data;
	rand_init(rd_data_, seed);
	alt_word_.resize(K_vocab);
	for (AliasTable &alt: alt_word_) {
		alt.Init(rd_data_, K_topic);
	}
}
DTM::UpdateWorker::~UpdateWorker () {
	delete rd_data_;
}

DTM::DTM (const Corpus &corpus, const Corpus &corpus_th, const Corpus &corpus_to, int N_gibbs_step,
		  double report_every, double dump_every, EpochSample *ctm) :
			N_gibbs_step(N_gibbs_step),
			report_every_(report_every), dump_every_(dump_every),
			K_topic(FLAGS_n_topics), K_vocab(corpus.K_vocab), K_doc_batch(FLAGS_n_doc_batch),
			vocab(corpus.vocab)
{
	assert(corpus.epochs.size() == corpus_th.epochs.size() &&
	       corpus.epochs.size() == corpus_to.epochs.size());

	omp_set_num_threads(FLAGS_n_threads);
	workers_.reserve(FLAGS_n_threads);

	const int primes[64] = {
		7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417,
		7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523,
		7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583, 7589, 7591,
		7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687, 7691, 7699,
		7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823,
		7829, 7841, 7853, 7867
	};
	std::random_device rd;
	for (int d = 0; d < FLAGS_n_threads; ++d) {
		if (FLAGS_fix_random_seed) {
			workers_.emplace_back(*this, primes[d]);
		}
		else {
			workers_.emplace_back(*this, rd());
		}
	}

	// Allocate and init training sample
	sample_[0].resize(corpus.epochs.size());
	for (int ep = 0; ep < sample_[0].size(); ++ep) {
		auto &s = sample_[0][ep];
		// Set up phi and alpha
		ArrayXXd phi, alpha;
		if (ctm) {
			alpha = ctm->alpha;
			phi = ctm->phi;
		}
		else {
			phi = ZEROS(K_topic, K_vocab);
			alpha = ZEROS(1, K_topic);
		}
		workers_[0].InitEpochSample(&s, phi, alpha, &corpus.epochs[ep]);
	}

	sample_[1].resize(corpus.epochs.size());
	for (int e = 0; e < sample_[1].size(); ++e) {
		workers_[0].InitEpochSample(&sample_[1][e], sample_[0][e].phi, sample_[0][e].alpha, sample_[0][e].corpus);
	}

	// Copy test corpus
	test_held_ = corpus_th.epochs;
	test_observed_ = corpus_to.epochs;
}

void SampleBatchId (vector<int> *dst, int n_total, int n_batch, rand_data *rd) {
	auto &d = *dst;
	d.resize(n_batch);
	for (int j = 0; j < n_batch; ++j) {
		for (bool repl = true; repl; ) {
			d[j] = irand(rd, 0, n_total);
			repl = false;
			for (int k = 0; k < j; ++k) {
				if ((repl = (d[k] == d[j])))
					break;
			}
		}
	}
}

void DTM::UpdateWorker::SampleBatch (EpochSample *st) {
	int n_batch = min(st->corpus->K_doc, K_doc_batch);
	SampleBatchId(&st->b.doc_id, st->corpus->K_doc, n_batch, rd_data_);
	sort(st->b.doc_id.begin(), st->b.doc_id.end()); // FIXME
}

void DTM::Infer () {
	TICK(last_report_time);
	TICK(last_dump_time);
 	cur_sample_idx_ = 0;
	for (int _ = 0; _ < N_gibbs_step; ++_) {
		auto &s_vec = sample_[0];

		// EVAULATION
		if (!_ || _ + 1 == N_gibbs_step || TOCK(last_report_time) > report_every_) {

			PRF(TICK(_ep_test));
			long double ppl = 0.;
#pragma omp parallel for reduction(+:ppl)
			for (int th = 0; th < FLAGS_n_threads; ++th) {
				for (int e = th; e < s_vec.size(); e += FLAGS_n_threads) {
					// ll_cond = (log) ...
					//   P(w_th | w_to, alpha, Phi) = P(w_th, w_to | alpha, Phi) / P(w_to | alpha, Phi)
					double ll_cur = workers_[th].EstimateLL(&s_vec[e], test_observed_[e], test_held_[e]);
					ppl += ll_cur / test_held_[e].K_tokens;
					LOG() << "perplexity of epoch " << e << ": "
						  << exp(-ll_cur / test_held_[e].K_tokens) << endl;
				}
			}
			ppl /= s_vec.size();
			PRF(LOG() << "Test time: " << TOCK(_ep_test) << "s\n");

			LOG() << "Round " << _ << " perplexity = " << exp(-ppl);
			LOG() << endl;
			if (FLAGS_show_topics) {
				for (int e = 0; e < s_vec.size(); ++e) {
					LOG() << "    Epoch " << e << ": \b";
					ShowTopics(&s_vec[e], 10);
					LOG() << s_vec[e].ct.transpose() << endl;
				}
			}

			last_report_time = NOW();
		}

		// Sample all alphas
		for (int e = 0; e < s_vec.size(); ++e) {
			auto &w = workers_[0];
			EpochSample *st = &s_vec[e];
			const EpochSample *stm1 = e ? &s_vec[e - 1] : nullptr;
			const EpochSample *stp1 = e + 1 < s_vec.size() ? &s_vec[e + 1] : nullptr;
			w.SampleBatch(st);
			w.UpdateZ(st, st);
			w.UpdateEta(st, st);
			w.UpdateAlpha(st, st, stm1, stp1);
			w.UpdatePhi(st, st, stm1, stp1);
		}
	}
}

void DTM::UpdateWorker::UpdateAlpha (EpochSample *st, const EpochSample *st0,
									 const EpochSample *stm1, const EpochSample *stp1)
{
	// I put a isotropic prior on \alpha_0. The expressions are modified accordingly.
	
	if (FLAGS_fix_alpha) {
		st->alpha *= 0;
		return;
	}
	
	int K_d = st0->corpus->K_doc;
	ArrayXXd mu, eta_bar;
	double var;

	eta_bar = st->eta.colwise().sum() / st->eta.rows();
	if (stp1 && stm1) {
		// Same as the paper
		ArrayXXd alpha_bar = 0.5 * (stm1->alpha + stp1->alpha);
		double s0 = 2 / sqr(stddev_al);
		double s1 = K_d / sqr(stddev_eta);
		mu = (s0 * alpha_bar + s1 * eta_bar) / (s0 + s1);
		var = 1.0 / (s0 + s1);
	}
	else if (stm1) {
		ArrayXXd alpha_bar = stm1->alpha;
		double s0 = 1 / sqr(stddev_al);
		double s1 = K_d / sqr(stddev_eta);
		mu = (s0 * alpha_bar + s1 * eta_bar) / (s0 + s1);
		var = 1.0 / (s0 + s1);
	}
	else if (stp1) {
		double sm1 = 1 / sqr(stddev_al0);
		double sp1 = 1 / sqr(stddev_al);
		double s_e = K_d / sqr(stddev_eta);
		mu = (/*sm1 * 0 +*/sp1 * stp1->alpha + s_e * eta_bar) / (sm1 + sp1 + s_e);
		var = 1.0 / (sm1 + sp1 + s_e);
	}
	else { // for debugging
		double sm1 = 1 / sqr(stddev_al0);
		double s_e = K_d / sqr(stddev_eta);
		mu = s_e * eta_bar / (sm1 + s_e);
		var = 1.0 / (sm1 + s_e);
	}
	SampleNormal(&st->alpha, mu, var, rd_data_);
	assert(mu.allFinite());
}

void DTM::UpdateWorker::UpdateEta (EpochSample *st, const EpochSample *st0) {
	auto K_doc = st0->corpus->K_doc;
	st->eta = st0->eta;

	for (int t = 0; t < FLAGS_n_sgld_e; ++t) {
		double eps = FLAGS_sgld_eta_a
			* (double)pow(FLAGS_sgld_eta_b + st->t_sgld_eta, -FLAGS_sgld_eta_c);
		st->t_sgld_eta += 1;
		// Note: If there were replication in batch the code would be incorrect.
		for (int i = 0; i < st0->b.size(); ++i) {
			int d = st0->b.doc_id[i];
			ArrayXXd eta_d = st->eta.row(d);
			// -1/\psi^2 * (eta^k_dt - \alpha^k_t)  % Note t[opic] in code is k in paper
			ArrayXXd gprior = -1.0 / sqr(stddev_eta) * (eta_d - st0->alpha);
			// + C^k_dt - (pi(eta_dt)_k) * N_dt)
			ArrayXXd gpost = st->cdt.row(i) - RowwSoftmax(eta_d) * st->nd.row(i).replicate(1, K_topic);
			ArrayXXd noise;
			SampleNormal(&noise, ZEROS(1, K_topic), eps, rd_data_);
			st->eta.row(d) += noise + eps / 2 * (gprior + gpost);
		}
		assert(st->eta.allFinite());
	}
}

void DTM::UpdateWorker::UpdatePhi (EpochSample *st, const EpochSample *st0,
								   const EpochSample *stm1, const EpochSample *stp1)
{
	auto K_doc = st0->corpus->K_doc;
	st->phi = st0->phi;
	for (int t = 0; t < FLAGS_n_sgld; ++t) {
		st->t_sgld_phi += 1;
		double eps_t = FLAGS_sgld_phi_a
			* (double)pow(FLAGS_sgld_phi_b + st->t_sgld_phi, -FLAGS_sgld_phi_c);
		ArrayXXd noise;
		SampleNormalBatch(&noise, ZEROS(1, K_vocab), eps_t, K_topic, rd_data_);
		// posterior = N_doc / N_batch * [C^w_kt - (C_kt * pi(Phi_kt)_w)]
		ArrayXXd g_post = st->cwt - (st->ct.replicate(1, K_vocab) * RowwSoftmax(st->phi));
		g_post = (double) st0->corpus->K_doc / st0->b.size() * g_post;
		// prior
		ArrayXXd g_prior = ZEROS(K_topic, K_vocab);
		if (stm1) {
			g_prior += (stm1->phi - st->phi) / sqr(stddev_phi);
		}
		else {
			g_prior += -st->phi / sqr(stddev_phi0);
		}
		if (stp1) {
			g_prior += (stp1->phi - st->phi) / sqr(stddev_phi);
		}
		st->phi += noise + eps_t / 2 * (g_post + g_prior);
	}
}

void DTM::UpdateWorker::BuildDocAlias (const EpochSample *s) {
	size_t K_doc_batch = s->b.size();
	alt_doc_.resize(K_doc_batch);
	for (size_t d = 0; d < K_doc_batch; ++d) {
		alt_doc_[d].Init(rd_data_, K_topic);
	}
	auto eta = s->eta;
	for (size_t d = 0; d < K_doc_batch; ++d) {
		alt_doc_[d].Rebuild(eta.row(s->b.doc_id[d]));
	}
}

void DTM::UpdateWorker::UpdateZ (EpochSample *st, const EpochSample *st0) {
	/*
	 * P(z|MB) \propto [exp(Phi_zw) / sum_w exp(Phi_zw)] * exp(Eta_dz)
	 *         =:      exp(log_pwt[z,w]) * exp(Eta_dz)
	 * (The paper seems to forget normalizing Phi; removing the normalization
	 *  result in slightly worse result)
	 */
	int K_doc_batch = (int)st0->b.size();
	auto eta = st0->eta;
	// ArrayXXd log_pwt = st0->phi; COMMENT-ME
	ArrayXXd log_pwt = LogRowwSoftmax(st0->phi);

	// Init alias tables. O((N_vocab + N_doc) * N_topic) - negligible
	BuildDocAlias(st0);
	for (int w = 0; w < K_vocab; ++w) {
		alt_word_[w].Rebuild(log_pwt.col(w));
	}

	st->cdt = ZEROS(K_doc_batch, K_topic);
	st->nd  = ZEROS(K_doc_batch, 1);
	st->cwt = ZEROS(K_topic, K_vocab);
	st->ct  = ZEROS(K_topic, 1);

	// cycled M-H. O(N_sum_doc_length_in_batch)
	for (int d_bpos = 0; d_bpos < K_doc_batch; ++d_bpos) {
		int d_id = st0->b.doc_id[d_bpos];
		const auto &tokens = st0->corpus->tokens[d_id];
		for (auto tok: tokens) {
			int w = tok.w;
			int z0 = alt_word_[w].Sample();
			for (int t = 0, _ = 0; t < tok.f; ++t) {
				for (int _steps = t ? 1 : FLAGS_n_mh_steps; _steps--; ++_) {
					int z1;
					double logA;
					if (!(_ & 1)) { // doc proposal
						z1 = alt_doc_[d_bpos].Sample();
						logA = log_pwt(z1, w) - log_pwt(z0, w);
					}
					else {
						z1 = alt_word_[w].Sample();
						logA = eta(d_id, z1) - eta(d_id, z0);
					}
					if (logA >= 0 || urand(rd_data_) < exp(logA)) {
						z0 = z1;
					}
				}
				st->cdt(d_bpos, z0) += 1;
				st->nd(d_bpos, 0) += 1;
				st->cwt(z0, w) += 1;
				st->ct(z0, 0) += 1;
			}
		}
	}
}

double DTM::UpdateWorker::EstimateLL (const EpochSample *tr, const EpCorpus &to, const EpCorpus &th) {
	/*
	 * Report P(w_th | \Phi, \alpha, w_to)
	 * = \int P(w_th|\Phi,\alpha,w_to,Z_th) P(Z_th|\Phi,\alpha,w_to) \dZ
	 * = \int P(w_th|[same]) P(Z_th|\Phi,\alpha,w_to,\eta) P(\eta|\Phi,\alpha,w_to) \dZ \deta.
	 * We draw samples from P(\eta|\Phi,\alpha,w_to) with Gibbs sampling to estimate the likelihood.
	 */
	assert(!tr->phi.hasNaN());
	Eigen::MatrixXd exp_phi = RowwSoftmax(tr->phi).matrix();

	// Sample eta ~ eta | Phi, alpha, w_to
	EpochSample tmp;
	InitEpochSample(&tmp, tr->phi, tr->alpha, &to);

	// Set up a full batch
	int K_doc_batch_bak = K_doc_batch;
	K_doc_batch = to.K_doc;
	SampleBatch(&tmp);
	K_doc_batch = K_doc_batch_bak;

	// Burn-in
	for (int _ = 0; _ < FLAGS_n_infer_burn_in; ++_) {
		UpdateZ(&tmp, &tmp);
		UpdateEta(&tmp, &tmp);
	}

	// Draw samples and estimate
	ArrayXXd lhoods = ZEROS(th.K_doc, FLAGS_n_infer_samples);
	for (int _ = 0; _ < FLAGS_n_infer_samples; ++_) {
		UpdateZ(&tmp, &tmp);
		UpdateEta(&tmp, &tmp);
		assert(!tmp.eta.hasNaN());
		Eigen::MatrixXd eta = RowwSoftmax(tmp.eta).matrix();
		for (int d = 0; d < th.K_doc; ++d) {
			for (auto &tok: th.tokens[d]) {
				lhoods(d, _) += tok.f * log(exp_phi.col(tok.w).dot(eta.row(d)));
			}
		}
	}

	auto sumLL = logsumexp(lhoods).sum() - log(FLAGS_n_infer_samples) * th.K_doc;
	return sumLL;
}

void DTM::ShowTopics (const EpochSample *st, int K) {
}

void DTM::DumpSample (string path) {
}

constexpr double
		DTM::stddev_al,
		DTM::stddev_eta,
		DTM::stddev_phi;
