cmake_minimum_required(VERSION 2.6)

aux_source_directory(. DIR_CLIPPER_SRCS)
add_library(clipper ${DIR_CLIPPER_SRCS})