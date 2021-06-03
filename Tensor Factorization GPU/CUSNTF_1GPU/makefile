ALS.exe:main.o dev.o
	nvcc -o SGD.exe main.o dev.o -lm -Wno-deprecated-gpu-targets -std=c++11
dev.o:dev.cu dev.h
	nvcc -c dev.cu -Wno-deprecated-gpu-targets -std=c++11
main.o:main.cu dev.h
	nvcc -c main.cu -Wno-deprecated-gpu-targets -std=c++11

chean:
	rm *.o -rf
