all: C/objs/image_processing_ispc.o
	rm *.so; python setup.py  build_ext --inplace

C/objs/image_processing_ispc.o: C/image_processing.ispc
	cd C; make clean; make;

.PHONY: all