#include <gflags/gflags.h>
#include <glog/logging.h>
#include "corpus.h"
#include "dtm.h"
#include "utils.h"
using namespace std;

DEFINE_string(train, "./data/nips.hb.tr.corpus", "training corpus");
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
DEFINE_double(report_every, 30, "Time in seconds between two consecutive reports");
DEFINE_double(dump_every, -1, "Time between dumps. <=0 -> never");
DECLARE_double(sgld_phi_a);
DECLARE_double(sgld_eta_a);

int main (int argc, char *argv[]) {
	
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	google::InitGoogleLogging(argv[0]);

	Corpus c_tr(FLAGS_train, FLAGS_dict);
	Corpus c_th(FLAGS_test_held, FLAGS_dict);
	Corpus c_to(FLAGS_test_observed, FLAGS_dict);
	assert(c_tr.K_vocab == c_to.K_vocab);

	if (FLAGS_trunc_input > 0) {
		c_tr.epochs.erase(c_tr.epochs.begin() + FLAGS_trunc_input, c_tr.epochs.end());
		c_th.epochs.erase(c_th.epochs.begin() + FLAGS_trunc_input, c_th.epochs.end());
		c_to.epochs.erase(c_to.epochs.begin() + FLAGS_trunc_input, c_to.epochs.end());
	}

	if (FLAGS_init_with_ctm) {
		LOG() << "Initializing single-epoch CTM ..." << endl;
		auto c_tr0 = Corpus::Merged(c_tr);
		auto c_th0 = Corpus::Merged(c_th);
		auto c_to0 = Corpus::Merged(c_to);
		DTM ctm(c_tr0, c_th0, c_to0, FLAGS_init_ctm_iter, FLAGS_report_every * 2, 0, nullptr);
		FLAGS_sgld_phi_a /= c_tr.epochs.size();
		FLAGS_sgld_eta_a /= c_tr.epochs.size();
		ctm.Infer();
		FLAGS_sgld_phi_a *= c_tr.epochs.size();
		FLAGS_sgld_eta_a *= c_tr.epochs.size();
		LOG() << "\nFitting DTM ..." << endl;
		DTM dtm(c_tr, c_th, c_to, FLAGS_n_iters, FLAGS_report_every, FLAGS_dump_every,
				&ctm.sample_[ctm.cur_sample_idx_][0]);
		dtm.Infer();
	}
	else {
		DTM dtm(c_tr, c_th, c_to, FLAGS_n_iters, FLAGS_report_every, FLAGS_dump_every, nullptr);
		dtm.Infer();
	}

	return 0;
}
