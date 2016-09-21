/*
 * TODO: to test
 * - Single proc single thread
 * - S proc multiple thread
 * - Single row m th
 * - Single col m th
 * - M m
 */
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "lcorpus.h"
#include "pdtm.h"
using namespace std;

DEFINE_string(corpus_prefix,
              "/home/dc/wkspace/btm_data/nips.hb",
              "prefix for corpus and dict");
DEFINE_string(test_held, "./data/nips.hb.th.corpus", "test corpus");
DEFINE_string(test_observed, "./data/nips.hb.to.corpus", "test corpus");
DEFINE_string(dict, "./data/nips.hb.dict", "dictionary file");
// DEFINE_string(train, "./data/tr.syn", "training corpus");
// DEFINE_string(test_held, "./data/th.syn", "test corpus");
// DEFINE_string(test_observed, "./data/to.syn", "test corpus");
// DEFINE_string(dict, "./data/dict.syn", "dictionary file");
DEFINE_string(log_path, "/tmp/dtm.last.log", "log path");
DEFINE_string(dump_prefix, "./last", "dump prefix");
DEFINE_int32(n_iters, 10000, "number of gibbs steps");
DEFINE_int32(init_ctm_iter, 200, "# gibbs steps for initialization");
DEFINE_int32(trunc_input, -1, "...");
DEFINE_bool(init_with_ctm, false, "initialize with a single diagonal CTM");
DECLARE_double(sgld_phi_a);
DECLARE_double(sgld_eta_a);

DEFINE_int32(n_vocab, 8000, "");
DEFINE_int32(proc_rows, 1, "");
DEFINE_int32(proc_cols, 1, "");

int main (int argc, char *argv[]) {
	
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

    // Init MPI and row comm
    int n_procs, proc_id;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    m_assert(n_procs == FLAGS_proc_rows * FLAGS_proc_cols);

    // TODO: load dict, th, to
    string corp_train = FLAGS_corpus_prefix + ".tr.corpus";
    string corp_theld = FLAGS_corpus_prefix + ".th.corpus";
    string corp_tobsv = FLAGS_corpus_prefix + ".to.corpus";
    string dict = FLAGS_corpus_prefix + ".dict";

    pDTM dtm(LocalCorpus(corp_train), FLAGS_n_vocab, proc_id, FLAGS_proc_rows, FLAGS_proc_cols);
	dtm.Infer();

	return 0;
}
