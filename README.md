# ZoneGraph

Requirements:
1. GDAL: https://gdal.org/index.html
2. GCC version 10+:  https://gcc.gnu.org/
3. OpenMP: https://www.openmp.org/
4. TBB: https://github.com/oneapi-src/oneTBB

Instructions for running the code:

1. Git clone the project
2. Download the dataset from https://drive.google.com/file/d/1IbZzC6C1ZODGQLmqnEiLkbNVCTbWFJM3/view?usp=sharing and put it in the same folder where you clone the code.
3. Compile the code using the provided Makefile: make
4. Run the code for TC1: ./run_tc1.sh
5. Output data are saved as raster files in ./Data/Results/TC1/Zone_Graph_v1/

Link to Ground Truth Labels: https://drive.google.com/drive/folders/1JR2g_9X8rVTl7UE93W6ITxSvrid03s5P?usp=sharing
Link to Baselines Data: https://drive.google.com/drive/folders/1QAgC8J7n2Qa2qrX8Rj1AWVYlkKaDBq3B?usp=sharing
