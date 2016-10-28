cmake_minimum_required(VERSION 3.5)
project(btm_dtm)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native \
-pthread -fopenmp  \
-DHAVE_SSE2 -DDSFMT_MEXP=19937")

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(BigTopicModel_SOURCE_DIR /home/if/wkspace/btm/src) # FIXME
include_directories(${BigTopicModel_SOURCE_DIR}/util)
include_directories(${BigTopicModel_SOURCE_DIR}/model)
include_directories(${BigTopicModel_SOURCE_DIR}/engine)

set(GLOG ${BigTopicModel_SOURCE_DIR}/../third_party/glog/build/libglog.a)
set(GFLAGS ${BigTopicModel_SOURCE_DIR}/../third_party/gflags/build/lib/libgflags.a)

include_directories(${BigTopicModel_SOURCE_DIR}/../third_party/glog/build)
include_directories(${BigTopicModel_SOURCE_DIR}/../third_party/glog/src)
include_directories(${BigTopicModel_SOURCE_DIR}/../third_party/gflags/build/include)

# add_library(glog STATIC IMPORTED)

set(SOURCE_FILES dSFMT/dSFMT.cpp
    main.cpp
    lcorpus.cpp
    lcorpus.h
    random.h aliastable.h
    utils.h pdtm.cpp pdtm.h mpi_circular.h)

add_executable(btm_dtm ${SOURCE_FILES})
target_link_libraries(btm_dtm ${MPI_CXX_LIBRARIES} ${GLOG} ${GFLAGS})

add_executable(dtm_fmt dtm_formatter.cpp)
target_link_libraries(dtm_fmt ${GLOG} ${GFLAGS})
