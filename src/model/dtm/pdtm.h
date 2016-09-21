//
// Created by w on 9/19/2016.
//

#ifndef BTM_DTM_PDTM_H
#define BTM_DTM_PDTM_H

#include <thread>
#include <mpi.h>
#include "lcorpus.h"
#include "random.h"
#include "dcm.h"
#include "aliastable.h"

class pDTM {
    static const int MAX_THREADS = 64;

    // TODO: Init stuff below
	MPI_Comm commRow;
	int procId, nProcRows, nProcCols, pRowId, pColId;
	int N_glob_vocab, N_topics, N_batch;

    vector<thread> threads;

	vector<size_t> nEpDocs;
	LocalCorpus c_train, c_test_observed, c_test_held;
	vector<Arr> localPhi;
	Arr localPhiZ;
	Arr phiTm1, phiTp1;
    vector<Arr> localPhiNormalized;
	vector<Arr> globEta;
	Arr sumEta, alpha;
    // Sampling eta requires same random settings in a row
	vector<rand_data> rd_data, rd_data_eta;
    // TODO: Init stuff above.

    struct BatchState {
        BatchState(LocalCorpus &corpus, int n_max_batch, pDTM &par);
        int N_glob_vocab, N_topics;
        pDTM &p;
        DCMSparse cdk;
        vector<Arr> cwk;
        Arr localEta;
        vector<pair<int, size_t>> localBatch, globalBatch;
        vector<vector<AliasTable>> altWord;
        LocalCorpus &corpus;
        inline void divideBatch();
        void UpdateZ_th(int thId, int nTh);
        void UpdateZ();
        void UpdateEta_th(int th, int nTh);
        void UpdateEta();
        void InitZ();
    } b_train, b_test;

	int iter;

	void IterInit(int t);
	void UpdatePhi(int th, int nTh);
	void UpdateAlpha();
    void EstimateLL();

public:

	pDTM(LocalCorpus &&c_train, LocalCorpus &&c_test_held, LocalCorpus &&c_test_observed, int n_vocab_,
             int procId_, int nProcRows_, int nProcCols_);

	void Infer();
};

#endif //BTM_DTM_PDTM_H
