#!/bin/bash

#get and enter current path
CUR=`dirname "$0"`
CUR=`cd "$CUR"; pwd`

#export CXX=icpc
#export CC=icc
export CXX=g++
export CC=gcc
#export CMAKE_PREFIX_PATH=/apps/mpi/mvapich-2.1.7a-gnu/:~/local/bin/gcc

mkdir -p release
pushd release
cmake .. -DCMAKE_BUILD_TYPE=Release
popd

mkdir -p debug
pushd debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
popd
