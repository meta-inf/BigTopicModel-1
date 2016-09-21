//
// Created by w on 9/19/2016.
// TODO: test EstimateLL
//

#include "pdtm.h"

using namespace std;

DEFINE_bool(fix_random_seed, true, "Fix random seed for debugging");
DEFINE_bool(show_topics, false, "Display top 10 words in each topic");
DEFINE_int32(n_sgld_phi, 2, "number of sgld iterations for phi");
DEFINE_int32(n_sgld_eta, 4, "number of sgld iterations for eta");
DEFINE_int32(n_mh_steps, 16, "number of burn-in mh iterations for Z");
DEFINE_int32(n_mh_thin, 1, "number of burn-in mh iterations for Z");
DEFINE_int32(n_infer_burn_in, 16, "number of burn-in steps in test");
DEFINE_int32(n_infer_samples, 1, "number of samples used in test");
DEFINE_int32(n_threads, 1, "number of threads used");
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
DEFINE_int32(report_every, 30, "Time in iterations between two consecutive reports");
DEFINE_int32(dump_every, -1, "Time between dumps. <=0 -> never");

DECLARE_int32(n_iters);
DECLARE_string(dump_prefix);

#define ZEROS_LIKE(a) Arr::Zero(a.rows(), a.cols())

pDTM::BatchState::BatchState(LocalCorpus &corpus_, int n_max_batch, pDTM &p_):
    p(p_), corpus(corpus_),
    N_glob_vocab(p.N_glob_vocab), N_topics(p.N_topics),
    cdk(1, p.nProcCols, n_max_batch, p.N_topics, row_partition, p.nProcCols, p.procId, FLAGS_n_threads)
{
    // cwk
    size_t n_row_eps = corpus.docs.size();
    int n_col_vocab = corpus.vocab_e - corpus.vocab_s;
    cwk.resize(n_row_eps);
    for (auto &a: cwk) a = Arr::Zero(N_topics, n_col_vocab);

    localEta = Arr::Zero(n_max_batch, p.N_topics);
}

pDTM::pDTM(LocalCorpus &&c_train, LocalCorpus &&c_test_held, LocalCorpus &&c_test_observed, int N_vocab_,
           int procId_, int nProcRows_, int nProcCols_) :
//    cwk(nProcCols_, 1, N_topics * (corpus_.ep_e - corpus_.ep_s), N_vocab_, column_partition, nProcCols_, procId_, FLAGS_n_threads),
    procId(procId_), nProcRows(nProcRows_), nProcCols(nProcCols_),
    N_glob_vocab(N_vocab_), N_topics(FLAGS_n_topics), N_batch(FLAGS_n_doc_batch),
    threads(FLAGS_n_threads),
    c_train(c_train), c_test_held(c_test_held), c_test_observed(c_test_observed),
    b_train(this->c_train, N_batch * (c_train.ep_e - c_train.ep_s), *this),
    b_test(this->c_test_observed, (int)this->c_test_observed.sum_n_docs, *this)
{
    pRowId = procId / nProcCols;
    pColId = procId % nProcCols;
    MPI_Comm_split(MPI_COMM_WORLD, pRowId, pColId, &commRow);

    // FIXME: REMOVE ME
    int t;
    MPI_Comm_rank(commRow, &t);
    assert(t == procId % nProcCols);

    // localPhi
    size_t n_row_eps = c_train.docs.size();
    int n_col_vocab = c_train.vocab_e - c_train.vocab_s;
    localPhi.resize(n_row_eps);
    for (auto &a: localPhi) a = Arr::Zero(N_topics, n_col_vocab);

    // phiTm1, phiTp1
    phiTm1 = ZEROS_LIKE(localPhi[0]);
    phiTp1 = ZEROS_LIKE(localPhi[0]);

    // localPhiNormalized
    localPhiNormalized.resize(size_t(c_train.ep_e - c_train.ep_s));
    for (auto &arr: localPhiNormalized) {
        arr = Arr::Zero(localPhi[0].rows(), localPhi[0].cols());
    }

    // globEta
    globEta.resize(n_row_eps);
    for (int ep = c_train.ep_s; ep < c_train.ep_e; ++ep)
        globEta[ep] = Arr::Zero(c_train.docs[ep].size(), N_topics);

    // sumEta, alpha
    sumEta = Arr::Zero(n_row_eps, N_topics);
    alpha = Arr::Zero(n_row_eps + 2, N_topics);

    // rd_data
    srand(233 * (nProcCols * nProcRows) + procId);
    rd_data.resize((size_t)FLAGS_n_threads);
    for (auto &r: rd_data) rand_init(&r, rand());

    // rd_data_eta is the same inside a row
    srand(233 * nProcRows + pRowId);
    rd_data_eta.resize((size_t)FLAGS_n_threads);
    for (auto &r: rd_data_eta) rand_init(&r, rand());
}

void SampleBatchId (vector<int> *dst, int n_total, int n_batch, rand_data *rd) {
    // TODO: Optimize this now that we have ~10K docs.
    assert(n_total >= n_batch);
    auto &d = *dst;
    d.resize((size_t)n_batch);
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

inline size_t cva_row_sum(CVA<SpEntry>::Row &row) {
    size_t ret = 0;
    for (const auto &e: row) ret += e.v;
    return ret;
}
inline void softmax(const double *src, double *dst, int n) {
    double max = -1e100, sum = 0;
    for (int d = 0; d < n; ++d) if (src[d] > max) max = src[d];
    for (int d = 0; d < n; ++d) sum += exp(dst[d] = src[d] - max);
    for (int d = 0; d < n; ++d) dst[d] /= sum;
}
template <typename T>
inline void divide_interval(T s, T e, int k, int n, T &ls, T &le) {
    T len = (e - s + n - 1) / n;
    ls = s + len * k;
    le = min(s + len * (k + 1), e);
}

void pDTM::BatchState::divideBatch() {
    size_t b_s, b_e;
    divide_interval((size_t)0, globalBatch.size(), p.pColId, p.nProcCols, b_s, b_e);
    localBatch.clear();
    localBatch.insert(localBatch.end(), globalBatch.begin() + b_s, globalBatch.begin() + b_e);
}

// NOTE:
// - phi, eta includes {N_vocab-1}th column in storage to simplify other computations;
//   They are clamped to zero as we're using reduced-normal and not updated.

void pDTM::IterInit(int iter) {
    this->iter = iter;
    // {{{ Reduce rowwise normalizer for Phi
    Arr phi_exp_sum = Arr::Zero(localPhi.size(), N_topics);
    for (int i = 0; i < (int)localPhi.size(); ++i) {
        for (int j = 0; j < N_topics; ++j)
            for (int k = 0; k < c_train.vocab_e - c_train.vocab_s; ++k)
                phi_exp_sum(i, j) += exp(localPhi[i](j, k));
    }
    localPhiZ = Arr::Zero(localPhi.size(), N_topics);
    MPI_Allreduce(phi_exp_sum.data(), localPhiZ.data(),
            eig_size(phi_exp_sum), MPI_DOUBLE, MPI_SUM, commRow);
    for (int i = 0; i < localPhiZ.rows(); ++i)
        for (int j = 0; j < localPhiZ.cols(); ++j)
            localPhiZ(i, j) = (double)log(localPhiZ(i, j));
    // }}}

    // {{{ Exchange PhiTm1, PhiTp1
    auto exTm1 = [&]() {
        if (pRowId == 0) return;
        MPI_Status status;
        int e = MPI_Sendrecv((void*)localPhi[0].data(), eig_size(localPhi[0]),
                MPI_DOUBLE, procId - nProcCols, iter * 4,
                (void*)phiTm1.data(), eig_size(phiTm1),
                MPI_DOUBLE, procId - nProcCols, iter * 4 + 1,
                MPI_COMM_WORLD, &status);
        assert(e == MPI_SUCCESS);
    };
    auto exTp1 = [&]() {
        if (pRowId == nProcRows - 1) return;
        MPI_Status status;
        int e = MPI_Sendrecv((void*)localPhi.back().data(), eig_size(localPhi[0]),
                MPI_DOUBLE, procId + nProcCols, iter * 4 + 1,
                (void*)phiTp1.data(), eig_size(phiTp1),
                MPI_DOUBLE, procId + nProcCols, iter * 4,
                MPI_COMM_WORLD, &status);
        assert(e == MPI_SUCCESS);
    };
    // Break blocking chain
    if (pRowId & 1) {
        exTp1(); exTm1();
    }
    else {
        exTm1(); exTp1();
    }
    // }}}

    // {{{ Sample batches
    vector<int> buff;
    buff.resize((size_t)2 * N_batch * b_train.corpus.docs.size());
    if (pColId == 0) {
        for (int i = 0, j = 0; i < b_train.corpus.docs.size(); ++i) {
            vector<int> tmp;
            SampleBatchId(&tmp, (int)b_train.corpus.docs[i].size(), N_batch, &rd_data[0]);
            for (int d: tmp) {
                buff[j++] = i;
                buff[j++] = d;
            }
        }
    }
    MPI_Bcast((void*)buff.data(), (int)buff.size(), MPI_INT, 0, commRow);
    b_train.divideBatch();
    // }}}

    // {{{ Normalize Phi
    auto worker = [this](int kTh, int nTh) {
        for (int e = 0; e < c_train.ep_e - c_train.ep_s; ++e)
            for (int v = c_train.vocab_s; v < c_train.vocab_e; ++v) {
                if (v % nTh != kTh) continue;
                int v_rel = v - c_train.vocab_s;
                for (int k = 0; k < N_topics; ++k) {
                    localPhiNormalized[e](k, v_rel) = localPhi[e](k, v_rel) - localPhiZ(e, k);
                }
            }
    };
    for (int t = 0; t < FLAGS_n_threads; ++t) {
        threads[t] = thread(worker, t, FLAGS_n_threads);
    }
    for (auto &th: threads) th.join();
    // }}}

#ifndef DTM_NDEBUG
    // Ensure rd_data_eta is unchanged
    string hbuf(sizeof(rand_data) * rd_data_eta.size(), '\0');
    std::copy((char*)rd_data_eta.data(), (char*)(rd_data_eta.data() + rd_data_eta.size()), hbuf.begin());
    uint64_t myHash = hash<string>()(hbuf), shHash = myHash;
    MPI_Bcast(&shHash, 1, MPI_INT64_T, 0, commRow);
    m_assert(shHash == myHash);
#endif
}

// UpdateZ and updateEta won't change persistent state (sample points etc.)
void pDTM::BatchState::UpdateZ() {
    InitZ();

    for (int _ = 0; _ < FLAGS_n_threads; ++_)
        p.threads[_] = thread(&pDTM::BatchState::UpdateZ_th, this, _, FLAGS_n_threads);
    for (int _ = 0; _ < FLAGS_n_threads; ++_)
        p.threads[_].join();

    cdk.sync();
    // cwk.sync();
}

void pDTM::BatchState::UpdateEta() {
    for (int _ = 0; _ < FLAGS_n_threads; ++_)
        p.threads[_] = thread(&pDTM::BatchState::UpdateEta, this, _, FLAGS_n_threads);
    for (int _ = 0; _ < FLAGS_n_threads; ++_)
        p.threads[_].join();
}

void pDTM::Infer() {
    for (int t = 0; t < FLAGS_n_iters; ++t) {
        IterInit(t);

        if (t % FLAGS_report_every == 0) {
            EstimateLL();
        }

        // Z
        b_train.UpdateZ();

        // Eta
        for (size_t di = 0; di < b_train.globalBatch.size(); ++di) {
            // initialize b_train.localEta
            b_train.localEta.row(di) = globEta[b_train.globalBatch[di].first].row(b_train.globalBatch[di].second);
            m_assert(! b_train.localEta.row(di).hasNaN()); // FIXME
        }
        b_train.UpdateEta();
        for (int d = 0, ep, rank; d < b_train.globalBatch.size(); ++d) {
            // commit changes for eta
            std::tie(ep, rank) = b_train.globalBatch[d];
            sumEta.row(ep) += b_train.localEta.row(d) - globEta[ep].row(rank);
            globEta[ep].row(rank) = b_train.localEta.row(d);
        }

        // Phi
        for (int _ = 0; _ + 1 < FLAGS_n_threads; ++_)
            threads[_] = thread(&pDTM::UpdatePhi, this, _, FLAGS_n_threads - 1);
        for (int _ = 0; _ < FLAGS_n_threads; ++_)
            threads[_].join();

        // Alpha
        if (0 == pColId) {
            UpdateAlpha();
        }
        MPI_Barrier(commRow);
    }
}

void pDTM::UpdateAlpha() {
    // Request alphaT{pm}1
    auto excM1 = [this] () {
        if (pRowId == 0) return; // Initialized to 0 as expected
        MPI_Status status;
        const double *send_data = alpha.data() + N_topics;
        double *recv_data = alpha.data();
        int r = MPI_Sendrecv(
                send_data, N_topics, MPI_DOUBLE, procId - nProcCols, iter * 4 + 2,
                recv_data, N_topics, MPI_DOUBLE, procId - nProcCols, iter * 4 + 3,
                MPI_COMM_WORLD, &status);
        assert(r == MPI_SUCCESS);
    };
    auto excP1 = [this] () {
        if (pRowId == nProcRows - 1) return;
        MPI_Status status;
        const double *send_data = alpha.data() + N_topics * (c_train.ep_e - c_train.ep_s);
        double *recv_data = alpha.data() + N_topics * (c_train.ep_e - c_train.ep_s + 1);
        int r = MPI_Sendrecv(
                send_data, N_topics, MPI_DOUBLE, procId + nProcCols, iter * 4 + 3,
                recv_data, N_topics, MPI_DOUBLE, procId + nProcCols, iter * 4 + 2,
                MPI_COMM_WORLD, &status);
        assert(r == MPI_SUCCESS);
    };
    if (pRowId & 1) {
        excP1(); excM1();
    }
    else {
        excM1(); excP1();
    }

    NormalDistribution normal;
    for (int ep = 0, ep_le = c_train.ep_e - c_train.ep_s; ep < ep_le; ++ep) {
        double *cur = alpha.data() + N_topics * (ep + 1);
        const double *pre = cur - N_topics, *nxt = cur + N_topics;
        // alpha_bar = (pre + nxt) / 2
        // s0 = 2 / sqr(FLAGS_sig_al)
        // s1 = N_docs / sqr(FLAGS_sig_eta)
        // mu = (s0 * alpha_bar + s1 * eta_bar) / (s0 + s1)
        //    = ((pre + nxt) / sqr(FLAGS_sig_al) + sumEta / sqr(FLAGS_sig_eta)) / (s0 + s1)
        // var = 1. / (s0 + s1)
        double s_al = 1. / sqr(FLAGS_sig_al), s_eta = 1. / sqr(FLAGS_sig_eta);
        double var = 1. / (2 * s_al + c_train.docs[ep].size() * s_eta);
        double std = (double)sqrt(var);
        for (int i = 0; i + 1 < N_topics; ++i) {
            double mu = ((pre[i] + nxt[i]) * s_al + sumEta(ep, i) * s_eta) * var;
            cur[i] = normal(&rd_data[0]) * std + mu;
        }
    }
}

// Sample Eta|MB.
// Each thread samples for a subset of documents.
void pDTM::BatchState::UpdateEta_th(int kTh, int nTh) {
    // Init thread-local storage
    static vector<double> eta_softmax_[MAX_THREADS];
    if (eta_softmax_[kTh].size() < N_topics) {
        eta_softmax_[kTh].resize(N_topics, 0.);
    }
    double *eta_softmax = eta_softmax_[kTh].data();

    // Divide doc batch for local thread
    size_t b_s, b_e;
    divide_interval((size_t)0, globalBatch.size(), kTh, nTh, b_s, b_e);

    // do SGLD
    NormalDistribution normal;
    for (int _ = 0; _ < FLAGS_n_sgld_eta; ++_) {
        int t = _ + p.iter * FLAGS_n_sgld_eta;
        double eps = FLAGS_sgld_eta_a * (double)pow(FLAGS_sgld_eta_b + t, -FLAGS_sgld_eta_c);
        double sq_eps = (double)sqrt(eps);

        // Update all docs
        for (int di = (int)b_s; di < b_e; ++di) {
            int ep_di = globalBatch[di].first;
            double *eta = localEta.data() + (size_t)N_topics * di;
            auto cdk = this->cdk.row(di);
            size_t cd = cva_row_sum(cdk);
            softmax(eta, eta_softmax, N_topics);

            // g_prior = 1 / sqr(sig_eta) * (alpha_t - eta)
            // g_post = cdk - RowwSoftmax(eta_d) * sum(cdk)
            // eta_d += N(0, eps) + eps/2 * (g_prior + g_post)

            // 1. Accumulate all terms except cdk
            double inv_sig_eta2 = 1. / sqr(FLAGS_sig_eta);
            for (int i = 0; i + 1 < N_topics; ++i) { // Last term is clamped to 0
                double g_prior = inv_sig_eta2 * (p.alpha(ep_di+1, i) - eta[i]);
                double g_post2 = -eta_softmax[i] * cd;
                eta[i] += normal(&p.rd_data_eta[kTh]) * sq_eps + (eps / 2) * (g_prior + g_post2);
                if (isnan(eta[i])) {
                    assert(false);
                }
            }
            // 2. Accumulate cdk
            for (const auto &e: cdk) {
                if (e.k != N_topics - 1) eta[e.k] += eps / 2 * e.v;
            }
        }
    }
}

void pDTM::UpdatePhi(int kTh, int nTh) {
    // TODO: pSGLD

    // Get vocab subset to sample
    int v_s, v_e;
    divide_interval(c_train.vocab_s, c_train.vocab_e, kTh, nTh, v_s, v_e);
    if (v_e == N_glob_vocab - 1) --v_e;

    // Sample
    NormalDistribution normal;
    for (int ep_g = c_train.ep_s; ep_g < c_train.ep_e; ++ep_g) {
        int ep_r = ep_g - c_train.ep_s;
        for (int _ = 0; _ < FLAGS_n_sgld_phi; ++_) {
            int t = FLAGS_n_sgld_phi * iter + _;
            double eps = FLAGS_sgld_phi_a * (double)pow(FLAGS_sgld_phi_b + t, -FLAGS_sgld_phi_c);
            double sqrt_eps = (double)sqrt(eps);
            auto &phi = localPhi[ep_r];
            const auto &phiTm1 = (ep_r == 0) ? this->phiTm1 : localPhi[ep_r - 1];
            const auto &phiTp1 = (ep_g + 1 == c_train.ep_e) ? this->phiTp1 : localPhi[ep_r + 1];
            /* for Topic k:
             * g_post = (N_docs_ep / N_batch) * [cwk[k] - ck[k] * softmax(phi)]
			 * g_prior = (phiTm1 + phiTp1 - 2*phi) / sqr(sigma_phi) [k] (first and last ep has different priors)
             * phi += N(0, eps) + eps / 2 * (g_post + g_prior) */

            double K_post = (double)c_train.docs[ep_r].size() / N_batch;
            for (int k = 0; k < N_topics; ++k) {
                const auto &cwk_k = b_train.cwk[ep_r].row(k);
                double ck_k = cwk_k.sum(); // TODO: expensive
                for (int w_g = v_s; w_g < v_e; ++w_g) {
                    int w_r = w_g - c_train.vocab_s;
                    double post = K_post * (cwk_k[w_r] - ck_k * (double)exp(localPhiNormalized[ep_r](k, w_r)));
                    double prior = 0;
                    prior += (0 == ep_g) ?
                             (-phi(k, w_r) / sqr(FLAGS_sig_phi0)) :
                             ((phiTm1(k, w_r) - phi(k, w_r)) / sqr(FLAGS_sig_phi));
                    prior += (pRowId + 1 == nProcRows && ep_g + 1 == c_train.ep_e) ?
                             (-phi(k, w_r) / sqr(FLAGS_sig_phi0)) :
                             ((phiTp1(k, w_r) - phi(k, w_r)) / sqr(FLAGS_sig_phi));
                    phi(k, w_r) += normal(&rd_data[kTh]) * sqrt_eps + eps / 2 * (post + prior);
                }
            }
        }
    }
}

void pDTM::BatchState::InitZ() {
    if (altWord.empty()) {
        // First entrance. Allocate stuff.
        altWord.resize(size_t(corpus.ep_e - corpus.ep_s));
        for (auto &vec: altWord) {
            vec.resize(size_t(corpus.vocab_e - corpus.vocab_s));
            for (auto &a: vec)
                a.Init(N_topics);
        }
    }

    // Reset cwk (cdk is cleared in sync())
    for (auto &a: cwk) a *= 0;

    auto worker = [this](int kTh, int nTh) {
        for (int e = 0; e < corpus.ep_e - corpus.ep_s; ++e)
            for (int v = kTh; v < corpus.vocab_e - corpus.vocab_s; v += nTh)
                altWord[e][v].Rebuild(p.localPhiNormalized[e].col(v));
    };
    vector<thread> threads((size_t)FLAGS_n_threads);
    for (int t = 0; t < FLAGS_n_threads; ++t) {
        threads[t] = thread(worker, t, FLAGS_n_threads);
    }
    for (auto &th: threads) th.join();
}

// Sample Z|MB.
void pDTM::BatchState::UpdateZ_th(int thId, int nTh) {
    // Divide docs
    size_t th_batch_s, th_batch_e;
    divide_interval((size_t)0, globalBatch.size(), thId, nTh, th_batch_s, th_batch_e);

    // Init thread-local alias table
    static AliasTable alt_docs[MAX_THREADS];
    auto &alt_doc = alt_docs[thId];
    alt_doc.Init(N_topics);

    // Sample Z
    for (size_t batch_id = th_batch_s; batch_id < th_batch_e; ++batch_id) {
        int ep = localBatch[batch_id].first; // relative
        size_t rank = localBatch[batch_id].second;
        const auto &log_pwt = p.localPhiNormalized[ep];

        // Init doc proposal
        alt_doc.Rebuild(localEta.row(batch_id));

        // M-H
        for (const auto &tok: corpus.docs[ep][rank].tokens) {
            int w_rel = tok.w - corpus.vocab_s;
            assert(tok.w >= corpus.vocab_s && tok.w < corpus.vocab_e);

            int z0 = altWord[ep][w_rel].Sample(&p.rd_data[thId]);
            for (int t = 0, _ = 0; t < tok.f; ++t) {
                for (int _steps = t ? FLAGS_n_mh_thin : FLAGS_n_mh_steps; _steps--; ++_) {
                    int z1;
                    double logA;
                    if (!(_ & 1)) { // doc proposal
                        z1 = alt_doc.Sample(&p.rd_data[thId]);
                        logA = log_pwt(z1, w_rel) - log_pwt(z0, w_rel);
                    }
                    else { // word proposal
                        z1 = altWord[ep][w_rel].Sample(&p.rd_data[thId]);
                        logA = localEta(batch_id, z1) - localEta(batch_id, z0);
                    }
                    if (logA >= 0 || urand(&p.rd_data[thId]) < exp(logA)) {
                        z0 = z1;
                    }
                }
                cdk.update(thId, (int)batch_id, z0);
                cwk[ep](z0, w_rel) += 1.;
                // cwk.update(thId, (ep - corpus.ep_s) * N_topics + z0, tok.w);
            }
        }
    }
}

inline Arr logsumexp (const Arr &src) {
    // TODO: optimize this
	Arr log_src = src;
	Eigen::ArrayXd maxC = log_src.rowwise().maxCoeff();
	log_src.colwise() -= maxC;
	return Eigen::log(Eigen::exp(log_src).rowwise().sum()) + maxC;
}

void pDTM::EstimateLL() {
    MPI_Barrier(commRow);

    // Init b_test
    b_test.localEta *= 0;
    // Divide batch
    b_test.globalBatch.clear();
    for (int e = 0; e < c_train.ep_e - c_train.ep_s; ++e) {
        for (int d = 0; d < c_train.docs[e].size(); ++d)
            b_test.globalBatch.push_back(make_pair(e, d));
    }
    b_test.divideBatch();

    // Burn-in
    for (int i = 0; i < FLAGS_n_infer_burn_in; ++i) {
        b_test.UpdateZ();
        b_test.UpdateEta();
    }

    // Draw samples and estimate
    // TODO: persistent storage for this ?
    Arr lhoods = Arr::Zero(c_train.sum_n_docs, FLAGS_n_infer_samples);
    for (int _ = 0; _ < FLAGS_n_infer_samples; ++_) {
        b_test.UpdateZ();
        b_test.UpdateEta();
        assert(!b_test.localEta.hasNaN());
        // TODO: multi-thread for this
        int d_p = 0, ep = 0;
        for (const auto &v: c_test_held.docs) {
            for (const auto &d: v) {
                for (const auto &tok: d.tokens) {
                    const auto &phi = localPhiNormalized[ep].col(tok.w - c_test_held.vocab_s);
                    const auto &eta = b_test.localEta.row(d_p);
                    lhoods(d_p, _) += tok.f * log(phi.cwiseProduct(eta).sum());
                }
                ++d_p;
            }
            ++ep;
        }
    }

    assert(! lhoods.hasNaN());
    if (! lhoods.allFinite()) { // may contain -inf
        LOG(INFO) << "Perplexity in row = inf";
    }
    else {
        long double arr[2] = {logsumexp(lhoods).sum(), c_test_held.sum_tokens}, rArr[2];
        MPI_Allreduce(arr, rArr, 2, MPI_LONG_DOUBLE, MPI_SUM, commRow);
        long double logEvi = rArr[0] - log(FLAGS_n_infer_samples) * c_test_held.sum_n_docs;
        double ppl = (double) exp(-logEvi / rArr[1]);
        LOG(INFO) << "Perplexity in row = " << ppl << " for " << rArr[1] << " tokens.";
    }
}

