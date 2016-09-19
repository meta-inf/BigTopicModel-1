//
// Created by w on 9/19/2016.
//

#include "pdtm.h"
#include "aliastable.h"
#include <thread>
using namespace std;

DEFINE_bool(fix_random_seed, true, "Fix random seed for debugging");
DEFINE_bool(show_topics, false, "Display top 10 words in each topic");
DEFINE_int32(n_sgld, 2, "number of sgld iterations for phi");
DEFINE_int32(n_sgld_eta, 4, "number of sgld iterations for eta");
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

DECLARE_int32(n_iters);
DECLARE_string(dump_prefix);

#define ZEROS_LIKE(a) Arr::Zero(a.rows(), a.cols())

pDTM::pDTM(LocalCorpus &&corpus_, int N_vocab_, int procId_, int nProcRows_, int nProcCols_):
    cdk(1, nProcCols_, FLAGS_n_doc_batch * (corpus_.ep_e - corpus_.ep_s), N_topics, row_partition, nProcCols_, procId_, FLAGS_n_threads),
    cwk(nProcCols_, 1, N_topics * (corpus_.ep_e - corpus_.ep_s), N_vocab_, column_partition, nProcCols_, procId_, FLAGS_n_threads),
    procId(procId_), nProcRows(nProcRows_), nProcCols(nProcCols_),
    N_topics(FLAGS_n_topics), N_batch(FLAGS_n_doc_batch),
    corpus(corpus_)
{
    MPI_Comm_split(MPI_COMM_WORLD, procId / nProcCols, procId % nProcCols, &commRow);
    // FIXME: REMOVE ME
    int t;
    MPI_Comm_rank(commRow, &t);
    assert(t == procId % nProcCols);

    // TODO: nEpDocs

    // localPhi
    size_t n_row_eps = corpus.docs.size();
    int n_col_vocab = corpus.vocab_e - corpus.vocab_s;
    localPhi.resize(n_row_eps);
    for (auto &a: localPhi) a = Arr::Zero(N_topics, n_col_vocab);

    // phiTm1, phiTp1
    phiTm1 = ZEROS_LIKE(localPhi[0]);
    phiTp1 = ZEROS_LIKE(localPhi[0]);

    // globEta
    globEta.resize(n_row_eps);
    for (int ep = corpus.ep_s; ep < corpus.ep_e; ++ep) {
        globEta[ep] = Arr::Zero(corpus.docs[ep].size(), N_topics);
    }
    // sumEta, localEta, alpha
    sumEta = Arr::Zero(n_row_eps, N_topics);
    localEta = Arr::Zero(FLAGS_n_doc_batch * n_row_eps, N_topics);
    alpha = Arr::Zero(n_row_eps+2, N_topics);

    // rd_data
    rd_data.resize((size_t)FLAGS_n_threads);
    for (auto &r: rd_data)
        rand_init(&r, 233);
}

void SampleBatchId (vector<int> *dst, int n_total, int n_batch, rand_data *rd) {
    // TODO: Optimize this now that we have ~10K docs.
    assert(n_total >= n_batch);
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

// NOTE:
// - phi, eta includes {N_vocab-1}th column in storage to simplify other computations;
//   They are clamped to zero as we're using reduced-normal and not updated.

void pDTM::IterInit(int iter) {
    this->iter = iter;
    // {{{ Reduce rowwise normalizer for Phi
    Arr phi_exp_sum = Arr::Zero(localPhi.size(), N_topics);
    for (int i = 0; i < (int)localPhi.size(); ++i) {
        for (int j = 0; j < N_topics; ++j)
            for (int k = 0; k < corpus.vocab_e - corpus.vocab_s; ++k)
                phi_exp_sum(i, j) += exp(localPhi[i](j, k));
    }
    localPhiZ = Arr::Zero(localPhi.size(), N_topics);
    MPI_Allreduce(phi_exp_sum.data(), localPhiZ.data(),
            eig_size(phi_exp_sum), MPI_DOUBLE, MPI_SUM, commRow);
    for (int i = 0; i < localPhiZ.rows(); ++i)
        for (int j = 0; j < localPhiZ.cols(); ++j)
            localPhiZ(i, j) = log(localPhiZ(i, j));
    // }}}

    // {{{ Exchange PhiTm1, PhiTp1.
    auto exTm1 = [&]() {
        MPI_Status status;
        int e = MPI_Sendrecv((void*)localPhi[0].data(), eig_size(localPhi[0]),
                MPI_DOUBLE, procId - nProcCols, iter * 4,
                (void*)phiTm1.data(), eig_size(phiTm1),
                MPI_DOUBLE, procId - nProcCols, iter * 4 + 1,
                MPI_COMM_WORLD, &status);
        assert(e == MPI_SUCCESS);
    };
    auto exTp1 = [&]() {
        MPI_Status status;
        int e = MPI_Sendrecv((void*)localPhi.back().data(), eig_size(localPhi[0]),
                MPI_DOUBLE, procId + nProcCols, iter * 4 + 1,
                (void*)phiTm1.data(), eig_size(phiTp1),
                MPI_DOUBLE, procId + nProcCols, iter * 4,
                MPI_COMM_WORLD, &status);
        assert(e == MPI_SUCCESS);
    }
    // Break blocking chain
    int pRow = procId / nProcCols;
    if (pRow & 1) {
        if (pRow != nProcRows - 1) exTp1();
        exTm1();
    }
    else {
        if (pRow != 0) exTm1();
        exTp1();
    }
    // }}}

    // {{{ Sample batches
    vector<int> buff;
    buff.resize((size_t)N_batch * corpus.docs.size());
    int pCol = procId % nProcCols;
    if (pCol == 0) {
        for (int i = 0, j = 0; i < corpus.docs.size(); ++i) {
            vector<int> tmp;
            SampleBatchId(&tmp, (int)corpus.docs[i].size(), N_batch, &rd_data[0]);
            for (int d: tmp) {
                buff[j++] = i;
                buff[j++] = d;
            }
        }
    }
    MPI_Bcast((void*)buff.data(), buff.size(), MPI_INT, 0, commRow);
    globalBatch.clear();
    for (int i = 0; i < buff.size(); i += 2) {
        globalBatch.push_back(make_pair(buff[i], buff[i + 1]));
    }
    size_t blen = (globalBatch.size() + nProcCols - 1) / nProcCols;
    localBatch.clear();
    localBatch.insert(localBatch.end(), globalBatch.begin() + pCol * blen,
            globalBatch.begin() + min((pCol + 1) * blen, globalBatch.size()));
    // }}}
}

void pDTM::Infer() {
    vector<thread> threads((size_t)FLAGS_n_threads);
    vector<double> etaBuff(N_batch * size_t(corpus.ep_e - corpus.ep_s));
    for (int t = 0; t < FLAGS_n_iters; ++t) {
        IterInit(t);

        // Z
        InitWordAlias();
        for (int _ = 0; _ < FLAGS_n_threads; ++_)
            threads[_] = thread(UpdateZ, this, _, FLAGS_n_threads);
        for (int _ = 0; _ < FLAGS_n_threads; ++_)
            threads[_].join();

        cdk.sync();
        cwk.sync();

        // Eta
        for (int _ = 0; _ < FLAGS_n_threads; ++_)
            threads[_] = thread(UpdateEta, this, _, FLAGS_n_threads);
        for (int _ = 0; _ < FLAGS_n_threads; ++_)
            threads[_].join();

        MPI_Barrier(commRow);
        // Allgather eta for cur batch & update globEta. Can be done in parallel with SamplePhi.
        threads[FLAGS_n_threads - 1] = thread([&]() {
            vector<size_t> offs;
            MPIHelpers::Allgatherv<double>(commRow, nProcCols, procId % nProcCols, localEta.data(), offs, etaBuff);
            int eb_i = 0;
            for (const auto &d: globalBatch) {
                for (int i = 0; i < N_topics; ++i) {
                    double nv = etaBuff[eb_i++];
                    sumEta(d.first, i) += nv - globEta[d.first](d.second, i);
                    globEta[d.first](d.second, i) = nv;
                }
            }
        });

        // Phi
        for (int _ = 0; _ + 1 < FLAGS_n_threads; ++_)
            threads[_] = thread(UpdatePhi, this, _, FLAGS_n_threads - 1);
        for (int _ = 0; _ < FLAGS_n_threads; ++_)
            threads[_].join();

        // Alpha
        if (procId % nProcCols == 0) {
            UpdateAlpha();
        }
        MPI_Barrier(commRow);
    }
}

void pDTM::UpdateAlpha() {
    // Request alphaT{pm}1
    auto excM1 = [this] () {
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
        MPI_Status status;
        const double *send_data = alpha.data() + N_topics * (corpus.ep_e - corpus.ep_s);
        double *recv_data = alpha.data() + N_topics * (corpus.ep_e - corpus.ep_s + 1);
        int r = MPI_Sendrecv(
                send_data, N_topics, MPI_DOUBLE, procId + nProcCols, iter * 4 + 3,
                recv_data, N_topics, MPI_DOUBLE, procId + nProcCols, iter * 4 + 2,
                MPI_COMM_WORLD, &status);
        assert(r == MPI_SUCCESS);
    };
    int pRow = procId / nProcCols;
    if (pRow & 1) {
        excP1(); excM1();
    }
    else {
        excM1(); excP1();
    }

    NormalDistribution dist;
    for (int ep = 0, ep_le = corpus.ep_e - corpus.ep_s; ep < ep_le; ++ep) {
        double *cur = alpha.data() + N_topics * (ep + 1);
        double *pre = cur - N_topics, *nxt = cur + N_topics;
        // alpha_bar = (pre + nxt) / 2
        // s0 = 2 / sqr(FLAGS_sig_al)
        // s1 = N_docs / sqr(FLAGS_sig_eta)
        // mu = (s0 * alpha_bar + s1 * eta_bar) / (s0 + s1)
        //    = ((pre + nxt) / sqr(FLAGS_sig_al) + sumEta / sqr(FLAGS_sig_eta)) / (s0 + s1)
        // var = 1. / (s0 + s1)
        double s_al = 1. / sqr(FLAGS_sig_al), s_eta = 1. / sqr(FLAGS_sig_eta);
        double var = 1. / (2 * s_al + corpus.docs[ep].size() * s_eta);
        double std = (double)sqrt(var);
        for (int i = 0; i < N_topics; ++i) {
            double mu = ((pre[i] + nxt[i]) * s_al + sumEta(ep, i) * s_eta) * var;
            cur[i] = dist(&rd_data[0]) * std + mu;
        }
    }
}

inline size_t cva_row_sum(const CVA<SpEntry>::Row &row) {
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

// Sample Eta|MB.
// Each proc/thread samples for a subset of documents.
void pDTM::UpdateEta(int kTh, int nTh) {
    // Init thread-local storage
    static vector<double> eta_softmax_[MAX_THREADS];
    if (eta_softmax_[kTh].size() < N_topics) {
        eta_softmax_[kTh].resize(N_topics, 0.);
    }
    double *eta_softmax = eta_softmax_[kTh].data();

    // Divide doc batch for local thread
    size_t b_s, b_e;
    divide_interval((size_t)0, localBatch.size(), kTh, nTh, b_s, b_e);

    // Load localEta
    for (size_t di = b_s; di < b_e; ++di) {
        localEta.row(di) = globEta[localBatch[di].first].row(localBatch[di].second);
    }

    // do SGLD
    NormalDistribution normal;
    for (int _ = 0; _ < FLAGS_n_sgld_eta; ++_) {
        int t = _ + iter * FLAGS_n_sgld_eta;
        double eps = FLAGS_sgld_eta_a
                     * (double)pow(FLAGS_sgld_eta_b + t, -FLAGS_sgld_eta_c);
        double sq_eps = (double)sqrt(eps);

        // Update all docs
        for (int di = (int)b_s; di < b_e; ++di) {
            int ep_di = localBatch[di].first;
            double *eta = localEta.data() + di * N_topics;
            auto cdk = this->cdk.row(di);
            size_t cdk_sum = cva_row_sum(cdk);
            softmax(eta, eta_softmax, N_topics);

            // g_prior = 1 / sqr(sig_eta) * (alpha_t - eta)
            // g_post = cdk - RowwSoftmax(eta_d) * sum(cdk)
            // eta_d += N(0, eps) + eps/2 * (g_prior + g_post)

            // 1. Accumulate all terms except cdk
            double inv_sig_eta2 = 1. / sqr(FLAGS_sig_eta);
            for (int i = 0; i + 1 < N_topics; ++i) { // Last term is clamped to 0
                double g_prior = inv_sig_eta2 * (alpha(ep_di+1, i) - eta[i]);
                double g_post2 = -eta_softmax[i] * cdk_sum;
                eta[i] += normal(&rd_data[kTh]) * sq_eps + (eps / 2) * (g_prior + g_post2);
            }
            // 2. Accumulate cdk
            for (const auto &t: cdk) {
                if (t.k != N_topics - 1) eta[t.k] += eps / 2 * t.v;
            }
        }
    }
}

void pDTM::InitWordAlias() {
    if (altWord.empty()) {
        // First entrance. Allocate stuff.
        altWord.resize(size_t(corpus.ep_e - corpus.ep_s));
        for (auto &vec: altWord) {
            vec.resize(size_t(corpus.vocab_e - corpus.vocab_s));
            for (auto &a: vec)
                a.Init(N_topics);
        }
        localPhiNormalized.resize(size_t(corpus.ep_e - corpus.ep_s));
        for (auto &arr: localPhiNormalized) {
            arr = Arr::Zero(localPhi[0].rows(), localPhi[0].cols());
        }
    }

    auto genWorker = [this](int kTh, int nTh) {
        return [this, kTh, nTh]() {
            for (int e = corpus.ep_s; e < corpus.ep_e; ++e)
                for (int v = corpus.vocab_s; v < corpus.vocab_e; ++v) {
                    if (v % nTh != kTh) continue;
                    int v_rel = v - corpus.vocab_s;
                    for (int k = 0; k < N_topics; ++k) {
                        localPhiNormalized[e](k, v_rel) = localPhi[e](k, v_rel) - localPhiZ(e, k);
                    }
                    altWord[e][v_rel].Rebuild(&rd_data[kTh], localPhiNormalized[e].col(v_rel));
                }
        };
    };
    vector<thread> threads(FLAGS_n_threads);
    for (int t = 0; t < FLAGS_n_threads; ++t) {
        threads[t] = thread(genWorker(t, FLAGS_n_threads));
    }
    for (auto &th: threads) th.join();
}

// Sample Z|MB.
void pDTM::UpdateZ(int kTh, int nTh) {
    // Divide docs
    size_t th_batch_s, th_batch_e;
    divide_interval((size_t)0, globalBatch.size(), kTh, nTh, th_batch_s, th_batch_e);

    // Init thread-local alias table
    static AliasTable alt_docs[MAX_THREADS];
    auto &alt_doc = alt_docs[kTh];
    alt_doc.Init(N_topics);

    for (size_t batch_id = th_batch_s; batch_id < th_batch_e; ++batch_id) {
        int ep = localBatch[batch_id].first;
        size_t rank = localBatch[batch_id].second;
        const auto &log_pwt = localPhiNormalized[ep];

        // Init doc proposal
        alt_doc.Rebuild(&rd_data[kTh], localEta.row(batch_id));

        // M-H
        for (const auto &tok: corpus.docs[ep][rank].tokens) {
            int w_rel = tok.w - corpus.vocab_s;
            assert(tok.w >= corpus.vocab_s && tok.w < corpus.vocab_e);

            int z0 = altWord[ep][w_rel].Sample();
            for (int t = 0, _ = 0; t < tok.f; ++t) {
                for (int _steps = t ? FLAGS_n_mh_thin : FLAGS_n_mh_steps; _steps--; ++_) {
                    int z1;
                    double logA;
                    if (!(_ & 1)) { // doc proposal
                        z1 = alt_doc.Sample();
                        logA = log_pwt(z1, w_rel) - log_pwt(z0, w_rel);
                    }
                    else { // word proposal
                        z1 = altWord[ep][w_rel].Sample();
                        logA = localEta(batch_id, z1) - localEta(batch_id, z0);
                    }
                    if (logA >= 0 || urand(&rd_data[kTh]) < exp(logA)) {
                        z0 = z1;
                    }
                }
                cdk.update(kTh, (int)batch_id, z0);
                cwk.update(kTh, (ep - corpus.ep_s) * N_topics + z0, tok.w);
            }
        }
    }
}
