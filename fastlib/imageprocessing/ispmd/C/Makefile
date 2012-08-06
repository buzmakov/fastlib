EXAMPLE=image_processing
CPP_SRC=c_image_processing.cpp
ISPC_SRC=image_processing.ispc
ISPC_TARGETS=sse2,sse4,avx

include ../common.mk

#CXX=g++ -m64
#CXXFLAGS=-Iobjs/ -O3 -Wall -fPIC
#ISPC=./ispc
#ISPCFLAGS=-O3 --arch=x86-64 --target=sse2,sse4,avx --pic
#
#ISPC_OBJS= objs/image_processing_ispc.h objs/image_processing_ispc.o \
    #objs/image_processing_ispc_sse2.o objs/image_processing_ispc_sse4.o objs/image_processing_ispc_avx.o
#
#OBJS=objs/c_image_processing.o $(ISPC_OBJS)
#
#default: image_processing
#
#.PHONY: dirs clean
#.PRECIOUS: objs/image_processing_ispc.h
#
#dirs:
	#/bin/mkdir -p objs/
#
#clean:
	#/bin/rm -rf objs *~ image_processing
#
#image_processing: dirs  $(OBJS)
	#$(CXX) $(CXXFLAGS) -o $@ $(OBJS)
#
#objs/%.o: %.cpp
	#$(CXX) $< $(CXXFLAGS) -c -o $@
#
#objs/c_image_processing.o: objs/image_processing_ispc.h
#
#objs/%_ispc.h objs/%_ispc.o objs/%_ispc_sse2.o objs/%_ispc_sse4.o objs/%_ispc_avx.o: %.ispc
	#$(ISPC) $(ISPCFLAGS) $< -o objs/$*_ispc.o -h objs/$*_ispc.h
