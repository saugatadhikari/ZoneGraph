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

Baselines Data:
For HMT and HMCT-PP, the model predictions are saved as raster files (.tif) and put in the folders HMT_Predictions and HMCT-PP_Predictions respectively for all 4 test regions.  
For FCN, the model weights are put in the folder FCN_Weights_for_each_region. For R1, R2 and R3 we used single weight to get predictions for the entire test region. For R4, since the region is very large, we used 2 different model weights to get the predictions for different portion of the test region as shown below:

R4[4000:12402, 3000:10032] --> We used **R4_Weight_A.pth** to get prediction on this portion of R4  
R4[7000:12402, 10032:20064] --> We used **R4_Weight_B.pth** to get prediction on this portion of R4  
R4[9000:12402, 20064:30096] --> We used **R4_Weight_A.pth** to get prediction on this portion of R4  
R4[12402:14402, 10032:20064] --> We used **R4_Weight_A.pth** to get prediction on this portion of R4  
R4[12402:18402, 20064:30096] --> We used **R4_Weight_B.pth** to get prediction on this portion of R4  
R4[12402:19402, 30096:37096] --> We used **R4_Weight_B.pth** to get prediction on this portion of R4  
