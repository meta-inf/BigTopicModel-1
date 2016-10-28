#!/usr/bin/env python

import os
import sys
from os.path import dirname, join
import time
import subprocess

mpi_cmd = "mpirun -n 4"
#hosts = " -perhost 1 -host juncluster4,juncluster2,juncluster5,juncluster1"
hosts = ""

# TODO: is working directory the same across nodes?

params = [
('data_prefix', "../data/bing/test"),
('n_threads', 4),
('n_topics', 100),
('report_every', 1),
('trunc_input', -1), # FIXME
('n_sgld_phi', 3),
('n_sgld_eta', 6),
('fix_random_seed', 1),
('fix_alpha', 0),
('n_iters', 200),
('n_mh_thin', 3),
('psgld', 1),
('sig_phi', 0.11),
('sig_phi0', 8),
('sig_al', 0.70),
('sig_al0', 0.0448),
('sgld_eta_a', 0.4),
('sgld_eta_b', 100),
('sgld_eta_c', 0.8),
('sgld_phi_a', 5),
('sgld_phi_b', 100),
('sgld_phi_c', 0.6)]

cmd  = "OMP_NUM_THREADS=%d "%thread_size + mpi_cmd + hosts + " ./btm_dtm" 
for p, v in params:
    cmd += " -%s=%s" % (p, str(v))

print cmd
#os.system(cmd)
