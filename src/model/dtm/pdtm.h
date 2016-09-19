//
// Created by w on 9/19/2016.
//

#ifndef BTM_DTM_PDTM_H
#define BTM_DTM_PDTM_H

#include <mpi.h>
#include "lcorpus.h"
#include "random.h"
#include "dcm.h"
#include "aliastable.h"

class pDTM {
    static const int MAX_THREADS = 64;

    // TODO: Init stuff below
	MPI_Comm commRow;
	int procId, nProcRows, nProcCols;
	int N_topics, N_batch;

	vector<size_t> nEpDocs;
	LocalCorpus corpus;
	vector<Arr> localPhi;
	Arr localPhiZ;
	Arr phiTm1, phiTp1;
	vector<Arr> globEta;
	Arr sumEta, localEta, alpha;
    DCMSparse cdk, cwk;
	vector<rand_data> rd_data;
    // TODO: Init stuff above.

	vector<pair<int, size_t>> localBatch, globalBatch;
    vector<vector<AliasTable>> altWord;
    vector<Arr> localPhiNormalized;
	int iter;

	void IterInit(int t);
	void UpdateZ(int th, int nTh);
	void UpdateEta(int th, int nTh);
    void InitWordAlias();
	void UpdatePhi(int th, int nTh);
	void UpdateAlpha();

public:

	pDTM(LocalCorpus &&corpus_, int n_vocab_, int procId_, int nProcRows_, int nProcCols_);

	void Infer();
};

#endif //BTM_DTM_PDTM_H
