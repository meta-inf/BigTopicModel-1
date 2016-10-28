//
// Created by if on 16-10-16.
//

#ifndef BTM_DTM_MPI_CIRCULAR_H
#define BTM_DTM_MPI_CIRCULAR_H

#include <mpi.h>
#include "utils.h"

template <typename T>
void Circular(const T *blk_local_first, const T *blk_local_last, T *blk_tm1, T *blk_tp1, size_t size_,
              int, int proc_id_m1, int proc_id_p1, int code_)
{
    if (proc_id_m1 < 0 && proc_id_p1 < 0) {
        return;
    }
    int size = (int)size_ * sizeof(T);
    auto send = [size](const T *buf_out, T *buf_in, int recv_from, int send_to, int code) {
        MPI_Status status;
        int e;
        if (recv_from == -1) {
            e = MPI_Send(buf_out, size, MPI_CHAR, send_to, code, MPI_COMM_WORLD);
        }
        else if (send_to == -1) {
            e = MPI_Recv(buf_in, size, MPI_CHAR, recv_from, code, MPI_COMM_WORLD, &status);
        }
        else {
            e = MPI_Sendrecv(buf_out, size, MPI_CHAR, send_to, code,
                             buf_in, size, MPI_CHAR, recv_from, code, MPI_COMM_WORLD, &status);
        }
        assert(e == MPI_SUCCESS);
    };
    send(blk_local_last, blk_tm1, proc_id_m1, proc_id_p1, code_);
    send(blk_local_first, blk_tp1, proc_id_p1, proc_id_m1, code_ + 1);
}

template <typename T>
void Circular_(const T *blk_local_first, const T *blk_local_last, T *blk_tm1, T *blk_tp1, size_t size_,
              int row_id, int proc_id_m1, int proc_id_p1, int code)
{
    int size = (int)size_ * sizeof(T);
    auto exTm1 = [&]() {
        if (proc_id_m1 < 0) return;
        MPI_Status status;
        int e = MPI_Sendrecv((void*)blk_local_first, size, MPI_CHAR, proc_id_m1, code,
                             (void*)blk_tm1, size, MPI_CHAR, proc_id_m1, code + 1,
                             MPI_COMM_WORLD, &status);
        assert(e == MPI_SUCCESS);
    };
    auto exTp1 = [&]() {
        if (proc_id_p1 < 0) return;
        MPI_Status status;
        int e = MPI_Sendrecv((void*)blk_local_last, size, MPI_CHAR, proc_id_p1, code + 1,
                             (void*)blk_tp1, size, MPI_CHAR, proc_id_p1, code,
                             MPI_COMM_WORLD, &status);
        assert(e == MPI_SUCCESS);
    };
    // Break blocking chain
    if (row_id & 1) {
        exTp1(); exTm1();
    }
    else {
        exTm1(); exTp1();
    }
}


#endif //BTM_DTM_MPI_CIRCULAR_H
