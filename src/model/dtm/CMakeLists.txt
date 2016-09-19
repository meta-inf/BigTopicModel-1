cmake_minimum_required(VERSION 3.5)
project(btm_dtm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(../../util)
include_directories(../../model)
include_directories(../../engine)

set(SOURCE_FILES
    lcorpus.cpp
    lcorpus.h
    random.h aliastable.h
    utils.h pdtm.cpp pdtm.h)

add_executable(btm_dtm ${SOURCE_FILES})