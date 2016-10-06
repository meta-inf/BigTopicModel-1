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
DEFINE_int32(n_sgld_e, 4, "number of sgld iterations for eta");
DEFINE_int32(n_mh_steps, 16, "number of burn-in mh iterations for Z");
DEFINE_int32(n_mh_thin, 1, "number of burn-in mh iterations for Z");
DEFINE_int32(n_infer_burn_in, 16, "number of burn-in steps in test");
DEFINE_int32(n_infer_samples, 1, "number of samples used in test");
DEFINE_int32(n_threads, 2, "number of threads used");
DEFINE_int32(n_topics, 50, "number of topics");
DEFINE_int32(n_doc_batch, 60, "implemented");
DEFINE_bool(psgld, false, "pSGLD with RMSProp for Phi");
DEFINE_double(psgld_a, 0.95, "alpha in RMSProp");
DEFINE_double(psgld_l, 1e-4, "lambda in pSGLD");
DEFINE_bool(sgld_mh, false, "Do MH test in SGLD");
DEFINE_bool(fix_alpha, false, "Use fixed symmetrical topic prior");
DEFINE_double(sgld_phi_a, 0.5, "");
DEFINE_double(sgld_phi_b, 100, "");
DEFINE_double(sgld_phi_c, 0.8, "");
DEFINE_double(sgld_eta_a, 0.5, "");
DEFINE_double(sgld_eta_b, 100, "");
DEFINE_double(sgld_eta_c, 0.8, "");
DEFINE_int32(sgld_mh_init, 100, "");
DEFINE_double(sig_al, 0.6, "");
DEFINE_double(sig_phi, 0.2, "");
DEFINE_double(sig_al0, 0.1, "");
DEFINE_double(sig_phi0, 10, "");
DEFINE_double(sig_eta, 4, "");

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
inline ArrayXXd RowwSoftmax (const ArrayXXd &src) { // Row-major; R^N => R^N
	ArrayXXd dst = src;
	dst.colwise() -= dst.rowwise().maxCoeff();
	dst = Eigen::exp(dst);
	dst.colwise() /= dst.rowwise().sum();
	return dst;
}

inline ArrayXXd Add0 (const ArrayXXd &src) { // Reduced-natural -> expaneded-natural
	ArrayXXd d1(src.rows(), src.cols() + 1);
	for (int d = 0; d < src.rows(); ++d) {
		for (int j = 0; j < src.cols(); ++j) {
			d1(d, j) = src(d, j);
		}
		d1(d, src.cols()) = 0;
	}
	return d1;
}
inline ArrayXXd Del0 (const ArrayXXd &src) { // e->R
	return src.block(0, 0, src.rows(), src.cols() - 1);
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
	s->phiAux = ZEROS(phi.rows(), phi.cols());
	s->t_sgld_eta = s->t_sgld_phi = 0;
}

DTM::UpdateWorker::UpdateWorker (DTM &p, int seed) :
	K_topic(p.K_topic), K_vocab(p.K_vocab), K_doc_batch(p.K_doc_batch), N_gibbs_step(p.N_gibbs_step),
	_sum_p_act(0), _sum_q_act(0)
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
// #define TESTS // FIXME_
#ifdef TESTS
	auto readMat = [&](const string &fileName, ArrayXXd &dest, int n, int m) {
		ifstream fin(fileName);
		dest = ZEROS(n, m);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				m_assert(!fin.eof());
				fin >> dest(i, j);
			}
		}
	};
	m_assert(FLAGS_n_topics==20);
	ArrayXXd phi_true[FLAGS_n_topics];
	for (int t = 0; t < FLAGS_n_topics; ++t) {
		char buf[100];
		sprintf(buf, "/home/dc/recycle_shift/dtm/dtm/dat/drun/lda-seq/topic-%03d-var-e-log-prob.dat", t);
		readMat(string(buf), phi_true[t], 8000, 13);
	}
#endif

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
			phi = ZEROS(K_topic, K_vocab - 1);
			alpha = ZEROS(1, K_topic - 1);
#ifdef TESTS
			for (int t = 0; t < K_topic; ++t) {
				for (int v = 0; v + 1 < K_vocab; ++v) {
					phi(t, v) = phi_true[t](v, ep) - phi_true[t](K_vocab-1, ep);
				}
			}
#endif
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
}

EpCorpus *FIXME_th, *FIXME_to;

void DTM::DebugPhi ()
{
	TICK(last_report_time);
	TICK(last_dump_time);
	auto &w = workers_[0];
	EpochSample *s_ref = &sample_[0][0]; // Contains true phi, eta and alpha by now
	EpochSample *s_cur = &sample_[1][0]; // Saves c** (Z-related things) and inferred phi
	s_cur->phi *= 0;

	cerr << "DEBUGGING PHI\n";

	for (int _ = 0; _ < N_gibbs_step; ++_) {
		if (!_ || _ + 1 == N_gibbs_step || TOCK(last_report_time) > report_every_) {
			PRF(TICK(_ep_test));
			long double ppl = 0.;
			double ll_cur = w.EstimateLL(s_cur, test_observed_[0], test_held_[0]);
			ppl += ll_cur / test_held_[0].K_tokens;
			ppl = exp(-ppl);
			if (ppl > 2 * K_vocab) {
				cerr << "Round " << _ << " perplexity = " << 1e20;
				break;
			}
			cerr << "Round " << _ << " perplexity = " << ppl;
			last_report_time = NOW();
		}

		w.SampleBatch(s_cur);
		w.UpdateZ(s_cur, s_cur);
		w.UpdatePhi(s_cur, s_cur, nullptr, nullptr);
	}
}

void DTM::Infer () {
	TICK(last_report_time);
	TICK(last_dump_time);
 	cur_sample_idx_ = 0;
	for (int _ = 0; _ < N_gibbs_step; ++_) {
		cur_sample_idx_ ^= 1;
		auto &s_cur = sample_[cur_sample_idx_];
		auto &s_pre = sample_[cur_sample_idx_ ^ 1];

		// EVAULATION
		if (!_ || _ + 1 == N_gibbs_step || TOCK(last_report_time) > report_every_) {
			PRF(TICK(_ep_test));
			long double ppl = 0.;
			long double ppl_train = 0;
#pragma omp parallel for reduction(+:ppl)
			for (int th = 0; th < FLAGS_n_threads; ++th) {
				for (int e = th; e < s_pre.size(); e += FLAGS_n_threads) {
					// ll_cond = (log) ...
					//   P(w_th | w_to, alpha, Phi) = P(w_th, w_to | alpha, Phi) / P(w_to | alpha, Phi)
					double ll_cur = workers_[th].EstimateLL(&s_pre[e], test_observed_[e], test_held_[e]);
					ppl += ll_cur / test_held_[e].K_tokens;
//#ifdef TESTS TODO
					double ll_train = workers_[th].EstimateLL(&s_pre[e], *s_pre[e].corpus, *s_pre[e].corpus);
					ppl_train += ll_train / s_pre[e].corpus->K_tokens;
					double ppl_train_ = exp(-ll_train / s_pre[e].corpus->K_tokens);
					cerr << "training ppl" << ppl_train_ << " ";
//#endif
					LOG() << "perplexity of epoch " << e << ": "
						  << exp(-ll_cur / test_held_[e].K_tokens) << endl;
				}
			}
			ppl /= s_pre.size();
			PRF(LOG() << "Test time: " << TOCK(_ep_test) << "s\n");
//#ifdef TESTS
			LOG() << "training " << " perplexity = " << exp(-ppl_train / s_pre.size());
//#endif

			ppl = exp(-ppl);
			if (ppl > 2 * K_vocab) {
				LOG() << "Round " << _ << " perplexity = " << 1e20;
				break;
			}
			LOG() << "Round " << _ << " perplexity = " << ppl;

#define MEAN(x) (x.sum()/x.rows()/x.cols())
			double mh_eta_act = workers_[0]._sum_q_act/workers_[0]._q_act.size();
			double mh_phi_act = workers_[0]._sum_p_act/workers_[0]._p_act.size();
			DBG_(mh_eta_act);
			DBG_(mh_phi_act);
			// DBG_(MEAN(Eigen::abs(s_pre[0].phi)));
			// DBG_(MEAN(Eigen::abs(s_pre[0].alpha)));
			LOG() << endl;
			if (FLAGS_show_topics) {
				for (int e = 0; e < s_pre.size(); ++e) {
					LOG() << "    Epoch " << e << ": \b";
					ShowTopics(&s_pre[e], 10);
					LOG() << s_pre[e].ct.transpose() << endl;
				}
			}

			if (dump_every_ > 0 && TOCK(last_dump_time) >= dump_every_) {
				DumpSample(FLAGS_dump_prefix);
				LOG() << "Sample dumped" << endl;
				last_dump_time = NOW();
			}

			last_report_time = NOW();
		}

		// 1 step of Gibbs sampling
#pragma omp parallel for
		for (int th = 0; th < FLAGS_n_threads; ++th) {
			auto &w = workers_[th];
			for (int e = th; e < s_cur.size(); e += FLAGS_n_threads) {
				EpochSample *st = &s_cur[e];
				EpochSample *st0 = &s_pre[e];
				//
				// FIXME_th = &test_held_[e];
				// FIXME_to = &test_observed_[e];
				//
				const EpochSample *stm1 = e ? &s_pre[e - 1] : nullptr;
				const EpochSample *stp1 = e + 1 < s_pre.size() ? &s_pre[e + 1] : nullptr;
				PRF(TICK(_ep_ctr));
				w.SampleBatch(st0);
				// do not change the order
				w.UpdateZ(st, st0);
				w.UpdateEta(st, st0);
				w.UpdateAlpha(st, st0, stm1, stp1);
				w.UpdatePhi(st, st0, stm1, stp1);
				// m_assert(st->eta.allFinite());
				// m_assert(st->alpha.allFinite());
				// m_assert(st->phi.allFinite());
				PRF(clog << TOCK(_ep_ctr) << endl);
			}
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
		double s0 = 2 / sqr(FLAGS_sig_al);
		double s1 = K_d / sqr(FLAGS_sig_eta);
		mu = (s0 * alpha_bar + s1 * eta_bar) / (s0 + s1);
		var = 1.0 / (s0 + s1);
	}
	else if (stm1) {
		ArrayXXd alpha_bar = stm1->alpha;
		double s0 = 1 / sqr(FLAGS_sig_al);
		double s1 = K_d / sqr(FLAGS_sig_eta);
		mu = (s0 * alpha_bar + s1 * eta_bar) / (s0 + s1);
		var = 1.0 / (s0 + s1);
	}
	else if (stp1) {
		double sm1 = 1 / sqr(FLAGS_sig_al0);
		double sp1 = 1 / sqr(FLAGS_sig_al);
		double s_e = K_d / sqr(FLAGS_sig_eta);
		mu = (/*sm1 * 0 +*/sp1 * stp1->alpha + s_e * eta_bar) / (sm1 + sp1 + s_e);
		var = 1.0 / (sm1 + sp1 + s_e);
	}
	else { // for debugging
		double sm1 = 1 / sqr(FLAGS_sig_al0);
		double s_e = K_d / sqr(FLAGS_sig_eta);
		mu = s_e * eta_bar / (sm1 + s_e);
		var = 1.0 / (sm1 + s_e);
	}
	SampleNormal(&st->alpha, mu, var, rd_data_);
	assert(mu.allFinite());
}

struct LDGrad {
	ArrayXXd k, ksum;
	ArrayXXd x, grad, softmax;
	double lhood, Kpost;
	int n;

	// P(x) \propto N(x|mu, sigma) \cdot (\prod_i \pi{x}_i^k_i) ^ Kpost

	LDGrad (const ArrayXXd &k, const ArrayXXd &ksum, int n, double Kpost = 1):
			k(k), ksum(ksum), n(n), Kpost(Kpost) {
		m_assert(k.rows() == 1 && k.cols() == n 
				&& ksum.rows() == 1 && ksum.cols() == 1);
	}

	inline void Bind (const ArrayXXd &x_, const ArrayXXd &mu, double stddev) {
		assert(x_.rows() == 1 && x_.cols() == n - 1 && mu.rows() == 1);
		x = x_;
		softmax = RowwSoftmax(Add0(x));
		ArrayXXd gprior = -1.0 / sqr(stddev) * (x_ - mu);
		ArrayXXd gpost = Kpost * Del0(k - softmax * ksum.replicate(1, n));
		grad = gprior + gpost;
		lhood = -1 / (2 * sqr(stddev)) * (x_ - mu).square().sum() +
				  Kpost * Eigen::log(softmax).cwiseProduct(k).sum(); 
	}

	inline double PT (const LDGrad &nxt, double eps) const {
		return -(nxt.x - x - eps / 2 * grad).square().sum() / (2 * eps);
	}

	inline double MH (const LDGrad &nxt, double eps) const {
		double ret = nxt.lhood - lhood + nxt.PT(*this, eps) - PT(nxt, eps);
		return ret;
	}
};

void DTM::UpdateWorker::UpdateEta (EpochSample *st, const EpochSample *st0) {
	auto K_doc = st0->corpus->K_doc;
	st->eta = st0->eta;

	if (FLAGS_sgld_mh) {
		double eps = FLAGS_sgld_eta_a
					 * (double)pow(FLAGS_sgld_eta_b + st->t_sgld_eta, -FLAGS_sgld_eta_c);
		st->t_sgld_eta += 1;
		double p_act = 0, p_tot = 0;
		for (int i = 0; i < st0->b.size(); ++i) {
			int d = st0->b.doc_id[i];
			LDGrad g[2] = {LDGrad(st->cdt.row(i), st->nd.row(i), K_topic),
						   LDGrad(st->cdt.row(i), st->nd.row(i), K_topic)};
			int cur = 0, nxt = 1;
			g[cur].Bind(st->eta.row(d), st->alpha, FLAGS_sig_eta);
			for (int _ = 0; _ < FLAGS_n_sgld_e; ++_) {
				ArrayXXd noise;
				SampleNormal(&noise, ZEROS(1, K_topic - 1), eps, rd_data_);
				g[nxt].Bind(g[cur].x + noise + eps / 2 * g[cur].grad, st->alpha, FLAGS_sig_eta);
				double logA = g[cur].MH(g[nxt], eps);
				if (logA >= 0 || urand(rd_data_) < exp(logA)) {
					swap(cur, nxt);
					p_act += 1;
				}
				p_tot += 1;
			}
			st->eta.row(d) = g[cur].x;
		}
		p_act /= p_tot;

		if (_q_act.size() < 50) {
			_sum_q_act += p_act;
			_q_act.push_back(p_act);
		}
		else {
			_sum_q_act += p_act - _q_act.back();
			_q_act.pop_back();
			_q_act.push_back(p_act);
		}
	}
	else {
		for (int t = 0; t < FLAGS_n_sgld_e; ++t) {
			double eps = FLAGS_sgld_eta_a
						 * (double)pow(FLAGS_sgld_eta_b + st->t_sgld_eta, -FLAGS_sgld_eta_c);
			st->t_sgld_eta += 1;
			// Note: If there were replication in batch the code would be incorrect.
			for (int i = 0; i < st0->b.size(); ++i) {
				int d = st0->b.doc_id[i];
				ArrayXXd eta_d = st->eta.row(d);
				// -1/\psi^2 * (eta^k_dt - \alpha^k_t)  % Note t[opic] in code is k in paper
				ArrayXXd gprior = -1.0 / sqr(FLAGS_sig_eta) * (eta_d - st0->alpha);
				// + C^k_dt - (pi(eta_dt)_k) * N_dt)
				ArrayXXd gpost = st->cdt.row(i) - RowwSoftmax(Add0(eta_d)) * st->nd.row(i).replicate(1, K_topic);
				ArrayXXd noise;
				SampleNormal(&noise, ZEROS(1, K_topic - 1), eps, rd_data_);
				st->eta.row(d) += noise + eps / 2 * (gprior + Del0(gpost));
			}
			assert(st->eta.allFinite());
		}
	}
}

static void pSgldUpdate (ArrayXXd &V, const ArrayXXd &grad, ArrayXXd &theta, double eps_t, rand_data *rd) {
	NormalDistribution normal_rnd_;
	V = FLAGS_psgld_a * V + (1 - FLAGS_psgld_a) * grad.square();
	// G = diag(1 / (lambda*1 + sqrt(V))). Note grad and theta are matrices instead of vecs.
	double avg_G = 0, maxg = 0;
	for (int r = 0; r < theta.rows(); ++r)
		for (int c = 0;c < theta.cols(); ++c) {
			double g = 1. / (FLAGS_psgld_l + (double)sqrt(V(r, c)));
			maxg = max(maxg, g);
			avg_G += g * g;
			double sig = (double)sqrt(eps_t * g);
			theta(r, c) += eps_t / 2 * g * grad(r, c) + sig * normal_rnd_(rd);
		}
}

void DTM::UpdateWorker::UpdatePhi (EpochSample *st, const EpochSample *st0,
								   const EpochSample *stm1, const EpochSample *stp1)
{
	auto K_doc = st0->corpus->K_doc;
	st->phi = st0->phi;
	st->phiAux = st0->phiAux;
	if (FLAGS_sgld_mh) {
		st->t_sgld_phi += 1;
		double eps_t = FLAGS_sgld_phi_a
					   * (double)pow(FLAGS_sgld_phi_b + st->t_sgld_phi, -FLAGS_sgld_phi_c);
		double p_act = 0, p_tot = 0;
		bool init = st->t_sgld_phi < FLAGS_sgld_mh_init;
		for (int t = 0; t < K_topic; ++t) {
			ArrayXXd mu; double stddev;
			if (stm1 && stp1) {
				mu = (stm1->phi.row(t) + stp1->phi.row(t)) / 2;
				stddev = FLAGS_sig_phi / sqrt(2);
			}
			else if (stm1) {
				mu = stm1->phi.row(t);
				stddev = FLAGS_sig_phi;
			}
			else if (stp1) {
				stddev = 1 / (1 / sqr(FLAGS_sig_phi) + 1 / sqr(FLAGS_sig_phi0));
				mu = stp1->phi.row(t) * (stddev / sqr(FLAGS_sig_phi));
				stddev = sqrt(stddev);
			}
			else {
				mu = ZEROS(1, K_vocab - 1);
				stddev = FLAGS_sig_phi0;
			}
			double Ksc = (double)K_doc / st0->b.size();
			LDGrad g[2] = {LDGrad(st->cwt.row(t), st->ct.row(t), K_vocab, Ksc),
						   LDGrad(st->cwt.row(t), st->ct.row(t), K_vocab, Ksc)};

			int cur = 0, nxt = 1;
			g[cur].Bind(st->phi.row(t), mu, stddev);
			int _p_loc = 0; // FIXME: debug
			for (int _ = init ? 1 : FLAGS_n_sgld; _--; ) {
				ArrayXXd noise;
				SampleNormal(&noise, ZEROS(1, K_vocab - 1), eps_t, rd_data_);
				g[nxt].Bind(g[cur].x + noise + eps_t / 2 * g[cur].grad, mu, stddev);
				double logA = g[cur].MH(g[nxt], eps_t);
				// We need the SGD-like behavior to move to a high prob region initially
				bool accept = logA >= 0 || urand(rd_data_) < exp(logA);
				if (init || accept) {
					swap(cur, nxt);
					p_act += accept;
					_p_loc += accept; // FIXME
				}
				p_tot += 1;
			}
			st->phi.row(t) = g[cur].x;
		}
		p_act /= p_tot;
		if (_p_act.size() < 50) {
			_sum_p_act += p_act;
			_p_act.push_back(p_act);
		}
		else {
			_sum_p_act += p_act - _p_act.back();
			_p_act.pop_back();
			_p_act.push_back(p_act);
		}
	}
	else {
		for (int t = 0; t < FLAGS_n_sgld; ++t) {
			st->t_sgld_phi += 1;
			double eps_t = FLAGS_sgld_phi_a
						   * (double)pow(FLAGS_sgld_phi_b + st->t_sgld_phi, -FLAGS_sgld_phi_c);
			// posterior = N_doc / N_batch * [C^w_kt - (C_kt * pi(Phi_kt)_w)]
			ArrayXXd g_post = st->cwt - (st->ct.replicate(1, K_vocab) * RowwSoftmax(Add0(st->phi)));
			g_post = (double) st0->corpus->K_doc / st0->b.size() * Del0(g_post);
			// prior
			ArrayXXd g_prior = ZEROS(K_topic, K_vocab - 1);
			if (stm1) {
				g_prior += (stm1->phi - st->phi) / sqr(FLAGS_sig_phi);
			}
			else {
				g_prior += -st->phi / sqr(FLAGS_sig_phi0);
			}
			if (stp1) {
				g_prior += (stp1->phi - st->phi) / sqr(FLAGS_sig_phi);
			}
			if (FLAGS_psgld) {
				g_post += g_prior;
				pSgldUpdate(st->phiAux, g_post, st->phi, eps_t, rd_data_);
			}
			else {
				ArrayXXd noise;
				SampleNormalBatch(&noise, ZEROS(1, K_vocab - 1), eps_t, K_topic, rd_data_);
				st->phi += noise + eps_t / 2 * (g_post + g_prior);
			}
		}
	}
}

void DTM::UpdateWorker::BuildDocAlias (const EpochSample *s) {
	size_t K_doc_batch = s->b.size();
	alt_doc_.resize(K_doc_batch);
	for (size_t d = 0; d < K_doc_batch; ++d) {
		alt_doc_[d].Init(rd_data_, K_topic);
	}
	auto eta = Add0(s->eta);
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
	auto eta = Add0(st0->eta);
	// ArrayXXd log_pwt = Add0(st0->phi); COMMENT-ME
	ArrayXXd log_pwt = LogRowwSoftmax(Add0(st0->phi));

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
/*
			AliasTable alt;
			alt.Init(rd_data_, K_topic);
			ArrayXd p_wd = ArrayXd(log_pwt.col(w)) + ArrayXd(eta.row(d_id));
			alt.Rebuild(p_wd);
*/
			int z0 = alt_word_[w].Sample();
			for (int t = 0, _ = 0; t < tok.f; ++t) {
				// z0 = alt.Sample();
				for (int _steps = t ? FLAGS_n_mh_thin : FLAGS_n_mh_steps; _steps--; ++_) {
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
	Eigen::MatrixXd exp_phi = RowwSoftmax(Add0(tr->phi)).matrix();

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
		Eigen::MatrixXd eta = RowwSoftmax(Add0(tmp.eta)).matrix();
		for (int d = 0; d < th.K_doc; ++d) {
			for (auto &tok: th.tokens[d]) {
				lhoods(d, _) += tok.f * log(exp_phi.col(tok.w).dot(eta.row(d)));
			}
		}
	}

	assert(! lhoods.hasNaN());
	if (! lhoods.allFinite()) { // may contain -inf
		return -1e100;
	}

	auto sumLL = logsumexp(lhoods).sum() - log(FLAGS_n_infer_samples) * th.K_doc;
	return sumLL;
}

void DTM::ShowTopics (const EpochSample *st, int K) {
	ArrayXXd phi_n = LogRowwSoftmax(Add0(st->phi));
	ArrayXXd alpha_n = LogRowwSoftmax(Add0(st->alpha));
	for (int t = 0; t < K_topic; ++t) {
		vector<pair<double, int>> ll(K_vocab);
		for (int j = 0; j < K_vocab; ++j) {
			ll[j] = make_pair(phi_n(t, j), j);
		}
		std::nth_element(ll.begin(), ll.end() - K, ll.end());
		std::sort(ll.end() - K, ll.end());
		LOG() << "\tTopic " << t << ":" << alpha_n(0, t) << ' ';
		for (int j = 0; j < K; ++j) {
			LOG() << vocab[ll[K_vocab - j - 1].second] << ':'
				  << ll[K_vocab - j - 1].first << ' ';
		}
		LOG() << endl;
	}
}

void DTM::DumpSample (string path) {
	if (path[path.size() - 1] != '/') {
		path += "/";
	}
	const auto &s = sample_[cur_sample_idx_ ^ 1]; // cur and pre have been swapped
	size_t K_epoch = s.size();
	vector<ArrayXXd> phi_ev(K_epoch);
	for (size_t d = 0; d < K_epoch; ++d) {
		phi_ev[d] = LogRowwSoftmax(Add0(s[d].phi));
	}
	for (int t = 0; t < K_topic; ++t) {
		char c_file[100];
		sprintf(c_file, "topic-%03d-var-e-log-prob.dat", t);
		ofstream fout(path + string(c_file));
		for (int v = 0; v < K_vocab; ++v) {
			for (size_t e = 0; e < K_epoch; ++e) {
				fout << setprecision(10) << phi_ev[e](t, v) << endl;
			}
		}
		fout.close();
	}
	ofstream f_alpha(path + "alpha.dat");
	for (size_t e = 0; e < K_epoch; ++e) {
		ArrayXXd alpha = LogRowwSoftmax(Add0(s[e].alpha));
		for (int t = 0; t < K_topic; ++t) {
			f_alpha << alpha(0, t) << " \n"[t + 1 == K_topic];
		}
	}
	f_alpha.close();
}

