#define cNum 2
#define LOGZERO -INFINITY
#define LOGLARGE -1000000
#define MAXGAIN INFINITY
#define MAX_ITERATIONS 15
#define LARGE_REGION_THRESHOLD 1000
#define UNIQUE_ELEV_THRESHOLD 1000
#define ADJ_PAIRS_THRESHOLD 400
#define MAX_COLOR_SET_SIZE 1024
// #define DEBUG
// #define AGG_N_SPLIT
// #define HMT_Tree

#include <iostream>
#include <functional>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <cmath>
#include <limits>
#include <cstdio>
#include <queue>
#include <stack>
#include <list>
#include <unordered_set>
#include <set>
#include <sstream>
#include <map>
#include <random>
#include "GeotiffRead.h"
#include "GeotiffWrite.h"
#include "DataTypes.h"

#include <tbb/tbb.h>
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include <tbb/parallel_sort.h>
#include <tbb/parallel_scan.h>
#include <tbb/mutex.h>
#include "omp.h"
#include <thread>
#include <tbb/concurrent_queue.h>
#include <mutex>

using namespace std;

typedef tbb::concurrent_queue<int> conque;

std::mutex mtx;

tbb::mutex floodCount;

omp_lock_t writelock;

// ------------------- CHECK
vector<Node> allNodes;
vector<int> bfsVisited;
vector<vector<int>> bfs;

conque que;

class cFlood {
private:
	struct Parameter parameter;
	struct Data data;
	vector<int>mappredictions;
	vector<int>mappredictions_plot;
	std::string HMFInputLocation;
	std::string HMFd8FlowDirection;
	std::string HMFProbability;
	std::string HMFFel;
	std::string HMFPara;
	std::string HMFStream;
	std::string HMFOutputFolderByDate;
	std::string HMFOutputLocation;
	std::string HMFPrediction;
	std::string HMFRGB;
	// std::string HMFLeftBank;
	std::string HMFPredictionRegularized;
	std::string HMFWeights;
	std::string HMFReg;
	std::string HMFNorm;
	int regType;
	int normalize;
	int nThreads;
	int nThreadsIntraZone;
	int nThreadsIntraZoneUB;
	int nFold; // number of fold for managing threads for nested parallelism
	int batch_size;
	int dynamic_batch_size;
	vector<int> node_location;
	vector<float> cost_map;

	//tree construction
	struct subset* subsets;

	int PIXELLIMIT;
	int largest_region_id;
	int second_largest_region_id;
	int largest_nei_id;
	int second_largest_nei_id;

public:
	void input(int argc, char* argv[]);
	void load_data();

	void UpdateTransProb(); //Update P(y|z), P(zn|zpn), P(zn|zpn=empty)

	//inference
	void inference();
	void updateMapPrediction();
	void prediction_D8_tree();
	void interpolate();
	void output();
	void prediction(string mode, float weight=0.0);

	vector<int> getBFSOrder(int root, vector<int>& bfsVisited, int region_id, bool flag);
	vector<int> getBFSOrderParallel(int root, vector<int>& bfsVisited, int region_id);
	void getBFSOrderParallelV2();
	static void bfsFunction(int region_id);
	static void bfsFunctionV2();
	vector<int> getSplitBFSOrder(int root, vector<int>& bfsVisited, int region_id);

	// hmt tree
	void HMTTree();
	void getNewBFSOrder();

	//utilities
	int find(struct subset subsets[], int i);
	void Union(struct subset subsets[], int x, int y);
	
	void validateTree();
	void validateTreeInference();
	void verifyHMTTree();

	// log likelihood regularization
	void getLoglikelihood(int region_idx);
	void getLoglikelihoodSerial(int region_idx);
	void getLoglikelihoodParallel(int region_idx);

	// region graph
	void adjacentRegions(int region_idx);

	// regularization
	void updateFrontierSerial(float weight);
	void updateFrontierParallel(float weight);
	void calcLambda(int region_id, int nei_idx);
	void updateEachFrontier(int region_idx, float weight);
	void getMinLoss(int region_idx, float weight);
	void getMinLossParallel(int region_idx, float weight);
	int getLambda(vector<AdjacentNodePair> adj_nodes, double currFloodLevel, double adjFloodLevel, int& lambda, int side);
	int getLambdaBot(vector<AdjacentNodePair> adj_nodes, double currFloodLevel);
	int getLambdaTop(vector<AdjacentNodePair> adj_nodes, double currFloodLevel);
	void computeRegTermSerial(int rIdx, int aIdx);
	void computeRegTermParallel(int rIdx, int aIdx);
	void computeRegTermWaterLevelDiff(int rIdx, int aIdx);
	void parallelScan(vector<int> input, vector<int>& output, int start, int end, int lambda_init);

	void parallelPrefixSum(vector<int>& input, vector<int>& output, int start, int end, int lambda_init);
	void serialPrefixSum(vector<int>& input, vector<int>& output, int start, int end, int lambda_init);

	static bool sortByZoneSize(RegionVertex a, RegionVertex b);
	static bool sortByAdjacent(AdjacentNodePair a, AdjacentNodePair b);
	static bool sortByCurrent(AdjacentNodePair a, AdjacentNodePair b);
	void insertAdjacent(vector<AdjacencyList> &adjacency_list, int current_node, int adjacent_node, int my_index, int neigh_region_id);

	int binarySearch(const vector<AdjacentNodePair> &adj_nodes, double targetCost);
	int binarySearchPair(const vector<AdjacentNodePair> &adj_nodes, double targetCost);
	pair<int, int> binarySearchMulti(const vector<AdjacentNodePair> &adj_nodes, double targetCost);
	int binarySearchOnSortedNodes(vector<Node> &sorted_nodes, double targetCost);

	void insertionSort(vector<AdjacentNodePair>& adjacentNodePairs);

	void coloring();
	void coloring_v2();

	size_t countViolatingPairs();

	// for creating sub-trees
	void aggNsplit(int region_idx);
	
	void writeRegionMap();
	void writeCostMap(vector<float> cost_map);
	void plot_colors();

	// debug codes, remove later !!!!!!!!!!!!!!!!!!!!!!!
	int getLambdaBotDebug(vector<AdjacentNodePair> adj_nodes, double currFloodLevel);
};



bool comp(Node a, Node b){
	return a.cost < b.cost;
}



// extended ln functions
double eexp(double x) {
	if (x == LOGZERO) {
		return 0;
	}
	else {
		return exp(x);
	}
}

double eln(double x) {
	if (x == 0) {
		return LOGZERO;
	}
	else if (x > 0) {
		return log(x);
	}
	else {
		std::cout << "Negative input error " << x << endl;
		exit(0);
	}
}


double eln_ll(double x) { 
	if (x == 0) {
		return LOGLARGE; // summation of multiple logzero (-inf) gives NaN value during LL calculation
	}
	else if (x > 0) {
		return log(x);
	}
	else {
		std::cout << "Negative input error " << x << endl;
		exit(0);
	}
}


double elnproduct(double x, double y) {
	// if (x == LOGZERO || y == LOGZERO) {
	// 	return LOGZERO;
	// }
	if (x == LOGZERO || y == LOGZERO) { // TODO: verify this
		return LOGLARGE;
	}
	else {
		return x + y;
	}
}
int dirExists(const char* const path)
{
	struct stat info;

	int statRC = stat(path, &info);
	if (statRC != 0)
	{
		if (errno == ENOENT) { return 0; } // something along the path does not exist
		if (errno == ENOTDIR) { return 0; } // something in path prefix is not a dir
		return -1;
	}

	return (info.st_mode & S_IFDIR) ? 1 : 0;
	// return (info.st_mode) ? 1 : 0;
}

void cFlood::insertAdjacent(vector<AdjacencyList> &adjacency_list, int current_node, int adjacent_node, int my_index, int neigh_region_id){
	AdjacencyList &new_region = adjacency_list[my_index];
	new_region.regionId = neigh_region_id;
	new_region.adjacentNodes.emplace_back(current_node, adjacent_node, data.allNodes[current_node].cost, data.allNodes[adjacent_node].cost);

	// new_region.adjacentNodes.emplace_back(AdjacentNodePair());
	// AdjacentNodePair &new_pair = new_region.adjacentNodes.back();

	// new_pair.currentNode = current_node;
	// new_pair.adjacentNode = adjacent_node;
	// new_pair.currNodeCost = data.allNodes[current_node].cost;
	// new_pair.adjNodeCost = data.allNodes[adjacent_node].cost;
}

bool cFlood::sortByZoneSize(RegionVertex a, RegionVertex b){
	int zone_size_a = a.regionSize;
	int zone_size_b = b.regionSize;

	return zone_size_a > zone_size_b;
}

// sort by adjacent region's elevation as primary key and current region's elevation as secondary key
bool cFlood::sortByAdjacent(AdjacentNodePair a, AdjacentNodePair b){
	double adj_node_cost_a = a.adjNodeCost;
	double adj_node_cost_b = b.adjNodeCost;

	if (adj_node_cost_a == adj_node_cost_b){
		return (a.currNodeCost < b.currNodeCost);
	}
	else{
		return (adj_node_cost_a < adj_node_cost_b);
	}	
}

// sort by current region's elevation as primary key
bool cFlood::sortByCurrent(AdjacentNodePair a, AdjacentNodePair b){
	return (a.currNodeCost < b.currNodeCost);
}



void cFlood::UpdateTransProb() {
	if (cNum != 2) {
		std::cout << "cannot handle more than two classes now!" << endl;
		std::exit(1);
	}

	double eln(double);
	// prior class probability of a node without parents
	// 2nd table on paper
	parameter.elnPz[0] = eln(1 - eexp(parameter.Pi)); // parameter.Pi is in log form here
	parameter.elnPz[1] = parameter.Pi;

	// class transitional probability
	// 1st table on paper
	parameter.elnPz_zpn[0][0] = eln(1);
	parameter.elnPz_zpn[0][1] = parameter.Epsilon; // epsilon is log(1-rho) on paper; input epsilon = 0.001, rho = close to 1
	parameter.elnPz_zpn[1][0] = eln(0);
	parameter.elnPz_zpn[1][1] = eln(1 - eexp(parameter.Epsilon));
	
	if (eexp(parameter.Epsilon) < 0 || eexp(parameter.Epsilon) > 1) {
		std::cout << "Epsilon Error: " << eexp(parameter.Epsilon) << endl;
	}
	if (eexp(parameter.Pi) < 0 || eexp(parameter.Pi) > 1) {
		std::cout << "Pi Error: " << eexp(parameter.Pi) << endl;
	}
	if (eexp(parameter.elnPz_zpn[0][1]) + eexp(parameter.elnPz_zpn[1][1]) != 1) {
		std::cout << "Error computing parameter.elnPz_zpn " << endl;
	}
	if (eexp(parameter.elnPz[0]) + eexp(parameter.elnPz[1]) != 1) {
		std::cout << "Error computing parameter.elnPz " << endl;
	}
}


vector<int> cFlood::getBFSOrder(int root, vector<int>& bfsVisited, int region_id, bool flag) {
	//vector<int> bfsVisited;
	vector<int> bfs;
	queue<int> que;
	que.push(root);

	double entry_point_elev = data.allNodes[root].elevation;

	while (!que.empty()) {
		int currentNode = que.front();
		bfs.push_back(currentNode);
		bfsVisited[currentNode] = 1;
		que.pop();

		if (flag){ // flag = 1 for initial BFS; for HMT no need to re-do this
			// assign region_id to each pixels
			if (currentNode == root)
				node_location[currentNode] = -2;
			else
				node_location[currentNode] = region_id;

			data.allNodes[currentNode].regionId = region_id;

			// calculate cost value of each pixel based on the elevation of entry point in the river
			double current_node_elev = data.allNodes[currentNode].elevation;
			data.allNodes[currentNode].cost = current_node_elev - entry_point_elev;
			cost_map[currentNode] = current_node_elev - entry_point_elev;
		}

		for (int i = 0; i < data.allNodes[currentNode].childrenID.size(); i++) {
			int child = data.allNodes[currentNode].childrenID[i];
			if (!bfsVisited[child]) {
				que.push(child);

			}
		}
		// for (int i = 0; i < data.allNodes[currentNode].parentsID.size(); i++) {
		// 	int parent = data.allNodes[currentNode].parentsID[i];
		// 	if (!bfsVisited[parent]) {
		// 		que.push(parent);
		// 	}
		// }
	}
	return bfs;
}

// void cFlood::bfsFunction(int region_id){
// 	while (true) {
//         int currentNode = -1;
//         if (!que[region_id].try_pop(currentNode)) {
//             // Queue is empty
//             break;
//         }

//         bfsVisited[currentNode] = true;

//         mtx.lock();
//         bfs[region_id].push_back(currentNode); // Store the visited node
//         mtx.unlock();

// 		for (int i = 0; i < allNodes[currentNode].childrenID.size(); i++) {
// 			int child = allNodes[currentNode].childrenID[i];
// 			if (!bfsVisited[child]) {
// 				que[region_id].push(child);
// 				bfsVisited[child] = true;
// 			}
// 		}
//     }
// }

void cFlood::bfsFunctionV2(){
	while (true) {
        int currentNode = -1;
        if (!que.try_pop(currentNode)) {
            // Queue is empty
            break;
        }

        bfsVisited[currentNode] = true;
		int region_id = allNodes[currentNode].regionId;

        mtx.lock();
        bfs[region_id].push_back(currentNode); // Store the visited node
        mtx.unlock();

		for (int i = 0; i < allNodes[currentNode].childrenID.size(); i++) {
			int child = allNodes[currentNode].childrenID[i];
			allNodes[child].regionId = region_id;
			if (!bfsVisited[child]) {
				que.push(child);
				bfsVisited[child] = true;
			}
		}
    }
}

// vector<int> cFlood::getBFSOrderParallel(int root, vector<int>& bfsVisited, int region_id) {

// 	//vector<int> bfsVisited;
// 	// vector<int> bfs;
// 	// conque que;
// 	que[region_id].push(root);

// 	double entry_point_elev = data.allNodes[root].elevation;

// 	std::vector<std::thread> thread_list;

// 	// Create and start threads
//     for (int i = 0; i < nThreadsIntraZoneUB; ++i) {
//         thread_list.push_back(std::thread(bfsFunction, region_id));
//     }

//     // Wait for all threads to finish
//     for (std::thread& t : thread_list) {
//         t.join();
//     }

// 	return bfs[region_id];
// }

void cFlood::getBFSOrderParallelV2() {
	std::vector<std::thread> thread_list;

	// Create and start threads
    for (int i = 0; i < nThreads; ++i) {
        thread_list.push_back(std::thread(bfsFunctionV2));
    }

    // Wait for all threads to finish
    for (std::thread& t : thread_list) {
        t.join();
    }
}

vector<int> cFlood::getSplitBFSOrder(int root, vector<int>& bfsVisited, int region_id) {
	//vector<int> bfsVisited;
	vector<int> bfs;
	queue<int> que;
	que.push(root);

	while (!que.empty()) {
		int currentNode = que.front();
		bfs.push_back(currentNode);
		bfsVisited[currentNode] = 1;
		que.pop();

		// assign region_id to each pixels
		if (currentNode == root)
			node_location[currentNode] = -2;
		else
			node_location[currentNode] = region_id;

		data.allNodes[currentNode].regionId = region_id;

		for (int i = 0; i < data.allNodes[currentNode].schildrenID.size(); i++) {
			int child = data.allNodes[currentNode].schildrenID[i];
			if (!bfsVisited[child]) {
				que.push(child);

			}
		}
		for (int i = 0; i < data.allNodes[currentNode].sparentsID.size(); i++) {
			int parent = data.allNodes[currentNode].sparentsID[i];
			if (!bfsVisited[parent]) {
				que.push(parent);
			}
		}
	}
	return bfs;
}

void cFlood::aggNsplit(int region_id){
	Regions& curr_region = data.allRegions[region_id];
	// parameter.split_threshold = 10000; // TODO: remove this

	// go in reverse of BFS order; this ensures than when I am checking current node, all its descendants are visited
	for (int i=curr_region.regionSize-1; i >= 0; i--){
		int node_id = curr_region.bfsOrder[i]; // reverse BFS order

		// entry point in the river always forms a region no matter what (BFS root node)
		if (node_id == curr_region.bfsRootNode){
			// cout << "node_id: " << node_id << endl;
			data.splitRegionIds.push_back(node_id);
			data.rootId2OrigRootId[node_id] = curr_region.bfsRootNode;
			continue;
		}

		// if enough nodes in sub-tree, split (this node_id forms a new sub-region)
		if (data.allNodes[node_id].stNodes >= parameter.split_threshold){
			data.splitRegionIds.push_back(node_id);
			data.rootId2OrigRootId[node_id] = curr_region.bfsRootNode;

			// root node of a sub-tree points to its original parent in global-tree
			data.allNodes[node_id].origParentID = data.allNodes[node_id].parentsID[0];
		}
		else{ // add my sub-tree node count to my parents
			int parent_id = data.allNodes[node_id].parentsID[0];
			data.allNodes[parent_id].stNodes += data.allNodes[node_id].stNodes;

			// add parent-child relation
			data.allNodes[node_id].sparentsID.push_back(parent_id);
			data.allNodes[parent_id].schildrenID.push_back(node_id);
		}
	}
}

void cFlood::load_data(){
	// reading raster files
	GeotiffRead d8FlowDirTiff((HMFInputLocation + HMFd8FlowDirection).c_str());
	GeotiffRead floodProbTiff((HMFInputLocation + HMFProbability).c_str());
	GeotiffRead elevationTiff((HMFInputLocation + HMFFel).c_str());
	GeotiffRead riverTiff((HMFInputLocation + HMFStream).c_str());
	GeotiffRead RGBTiff((HMFInputLocation + HMFRGB).c_str());

	float** d8FlowDirData = (float**)d8FlowDirTiff.GetRasterBand(1);
	float** floodProbData = floodProbTiff.GetRasterBand(1);
	float** elevationData = elevationTiff.GetRasterBand(1);
	float** riverData = riverTiff.GetRasterBand(1);
	float** rgbData = RGBTiff.GetRasterBand(1);

	// Get the array dimensions
	int* dims = d8FlowDirTiff.GetDimensions();

	parameter.ROW = dims[0];
	parameter.COLUMN = dims[1];
	int total_pixels = dims[0] * dims[1];
	parameter.allPixelSize = total_pixels;

	if (parameter.Epsilon > 1 || parameter.Pi > 1) {
		std::cout << "wrong parameter" << endl;
	}

	std::cout << "Input parameters: " << endl;
	cout << "Region: " << parameter.regionId << endl;
	cout << "Epsilon: " << parameter.Epsilon << " Pi: " << parameter.Pi << endl;
	cout << "rho: " << parameter.rho << endl;

	auto start_2 = std::chrono::steady_clock::now();

	data.allNodes.resize(total_pixels, Node());

	parameter.elnPzn_xn.resize(total_pixels * cNum, eln(0.5)); // stored as [dry, flood, dry, flood, dry, flood ...], acc. to old data

	vector<int> all_pixels(total_pixels);
    iota(all_pixels.begin(), all_pixels.end(), 0);

	// // Shuffle the vector using a random engine
    // std::random_device rd;
    // std::mt19937 rng(rd());
    // std::shuffle(all_pixels.begin(), all_pixels.end(), rng);

	#pragma omp parallel for schedule(static) num_threads(1)
	for (int idx=0; idx<total_pixels; idx++){
	// for (int row = 0; row < parameter.ROW; row++){
	// 	for (int col = 0; col < parameter.COLUMN; col++){
			int currentId = all_pixels[idx];

			int row = (int)(currentId / parameter.COLUMN);
			int col = currentId % parameter.COLUMN;

			int d8FlowDir = d8FlowDirData[row][col];
			float floodProb = floodProbData[row][col]; // prob from U-Net
			float elevation = elevationData[row][col];
			int riverNode = riverData[row][col]; // -1 means not in river; everything else is in river
			int rgb = rgbData[row][col];

			if (riverNode > 0) { // if -1 is not set, put >0 else >= 0
				// src_dir_m_1++;
				d8FlowDir = 0;
			}

			// int currentId = row * parameter.COLUMN + col;
			int parentId;
			bool river_node = false;

			// identify where does the water flow from current pixel --> parent direction
			switch (d8FlowDir) {
				case 0:
					#pragma omp critical
					{
						data.river_nodes.push_back(currentId);
						river_node = true;
					}
					break;
				case 1:
					parentId = currentId + 1;
					break;
				case 2:
					parentId = currentId - parameter.COLUMN + 1;
					break;
				case 3:
					parentId = currentId - parameter.COLUMN;
					break;
				case 4:
					parentId = currentId - parameter.COLUMN - 1;
					break;
				case 5:
					parentId = currentId - 1;
					break;
				case 6:
					parentId = currentId + parameter.COLUMN - 1;
					break;
				case 7:
					parentId = currentId + parameter.COLUMN;
					break;
				case 8:
					parentId = currentId + parameter.COLUMN + 1;
					break;
				default:
					// skipped_nodes++;
					break;
			}

			// if (d8FlowDir == 0){
			// 	#pragma omp critical
			// 	{
			// 		data.river_nodes.push_back(currentId);
			// 	}
			// 	river_node = true;
			// }
			// else{
			// 	parentId = currentId + (mult_map[d8FlowDir] * parameter.COLUMN) + add_map[d8FlowDir];
			// }

			data.allNodes[currentId].node_id = currentId;
			data.allNodes[currentId].isNa = (rgb == 0) ? true : false; 
			data.allNodes[currentId].p = floodProb;
			data.allNodes[currentId].elevation = elevation; // pit filled elevation layer

			if (river_node){
				// data.allNodes[currentId].p = 0.999;
				// parameter.elnPzn_xn[currentId * cNum] = eln(0.001);
				// parameter.elnPzn_xn[currentId * cNum + 1] = eln(0.999);

				data.allNodes[currentId].p = 1.0;
				parameter.elnPzn_xn[currentId * cNum] = -INFINITY; // eln(0.0);
				parameter.elnPzn_xn[currentId * cNum + 1] = 0; // eln(1.0);

				continue;
			}
			else if (data.allNodes[currentId].isNa){
				data.allNodes[currentId].p = 0.5;
				parameter.elnPzn_xn[currentId * cNum] = eln(0.5);
				parameter.elnPzn_xn[currentId * cNum + 1] = eln(0.5);
			}
			else{
				parameter.elnPzn_xn[currentId * cNum] = eln(1 - floodProb); // stored as [dry, flood, dry, flood, dry, flood ...]
				parameter.elnPzn_xn[currentId * cNum + 1] = eln(floodProb);
			}
			
			#pragma omp critical
			{
				data.allNodes[currentId].parentsID.push_back(parentId); // HMTTree has multiple parents

				if (parentId >= 0 && parentId < total_pixels){
					data.allNodes[parentId].childrenID.push_back(currentId);
				}
			}
		// }
	}

	auto end_2 = std::chrono::steady_clock::now();
	auto elapsed_seconds_2 = end_2 - start_2;
	std::cout << "Getting Parent and Children of each pixel: " << std::chrono::duration_cast<std::chrono::seconds>(elapsed_seconds_2).count() << " seconds" << endl << endl;

	//Free each sub-array
    for(int i = 0; i < parameter.ROW; ++i) {
        delete[] d8FlowDirData[i]; 
		delete[] floodProbData[i];
		delete[] elevationData[i];
		delete[] riverData[i];
		delete[] rgbData[i]; 
    }

    //Free the array of pointers
    delete[] d8FlowDirData;
	delete[] floodProbData;
	delete[] elevationData;
	delete[] riverData;
	delete[] rgbData;
}



void cFlood::input(int argc, char* argv[]) {
	GDALAllRegister();

	if (argc > 2) {
		ifstream config(argv[1]);
		string line;
		getline(config, line);
		HMFInputLocation = line; 
		getline(config, line);
		HMFd8FlowDirection = line;        
		getline(config, line);
		HMFProbability = line;
		getline(config, line);
		HMFFel = line;
		getline(config, line);
		HMFStream = line;
		getline(config, line);
		HMFRGB = line;
		getline(config, line);
		HMFPara = line;
		getline(config, line);
		HMFOutputFolderByDate = line;
		getline(config, line);
		HMFOutputLocation = line;
		getline(config, line);
		HMFPrediction = line;
		getline(config, line);
		HMFPredictionRegularized = line;
		getline(config, line);
		HMFWeights = line;
		getline(config, line);
		HMFReg = line;
		regType = stoi(HMFReg);
		getline(config, line);
		HMFNorm = line;
		normalize = stoi(HMFNorm);

		nThreads = atoi(argv[2]);
		nFold = atoi(argv[3]);
		batch_size = atoi(argv[4]);
		dynamic_batch_size = atoi(argv[5]);
	}
	else {
		std::cout << "Invalid Arguments. Please provide config file path, num threads, num threads intra zone and batch size!";
	}

	// nThreadsIntraZoneUB = min(nThreads, nThreadsIntraZone);
	// nThreadsIntraZoneUB = nThreadsIntraZone; // !!!!!!!!!!!!!!!!!!!

	
	if (nFold == 0){
		nThreadsIntraZoneUB = 1;
	}
	else{
		nThreadsIntraZoneUB = 32 * nFold / nThreads;
	}
	

	cout << "# folds: " << nFold << endl;
	cout << "# threads: " << nThreads << endl;
	cout << "# threads intra zone: " << nThreadsIntraZoneUB << endl;
	cout << "batch size: " << batch_size << endl;

	omp_set_num_threads(nThreads);
	omp_set_max_active_levels(4);
	omp_init_lock(&writelock);

	int status = dirExists(HMFInputLocation.c_str());
	if (status <= 0) {
		cout << "Error: input directory does not exist.." << endl;
		exit(0);
	}

	status = dirExists(HMFOutputFolderByDate.c_str());
	if (status <= 0) {
		cout << "Output folder by date does not exist..creating one!" << endl;
		status = mkdir(HMFOutputFolderByDate.c_str(), 0777); // create output dir if not exists
		if (status != 0) {
			cout << "Error: could not create Output Folder by date.." << endl;
			exit(0);
		}
	}

	status = dirExists(HMFOutputLocation.c_str());
	cout << " Output Location: " << HMFOutputLocation << endl;
	if (status <= 0) {
		cout << "Output directory does not exist..creating one!" << endl;
		status = mkdir(HMFOutputLocation.c_str(), 0777); // Create output dir if not exists
		if (status != 0) {
			cout << "Error: could not create Output Directory.." << endl;
			exit(0);
		}
	}

	// Added by Saugat: create directory to store loglikelihood and height
	status = dirExists((HMFOutputLocation + "LogLikelihood_Height").c_str());
	if (status <= 0) {
		status = mkdir((HMFOutputLocation + "LogLikelihood_Height").c_str(), 0777);
		if (status != 0) {
			cout << "Error: could not create LogLikelihood_Height folder.." << endl;
			exit(0);
		}
	}

	//reading text file
	ifstream parameterFile(HMFInputLocation + HMFPara);
	if (!parameterFile) {
		std::cout << "Failed to open parameter!" << endl;
		exit(0);
	}
	string line1;

	getline(parameterFile, line1);
	parameter.regionId = stoi(line1);
	getline(parameterFile, line1);
	parameter.Epsilon = stod(line1);
	getline(parameterFile, line1);
	parameter.Pi = stod(line1);
	getline(parameterFile, line1);
	parameter.lambda = stof(line1);
	getline(parameterFile, line1);
	PIXELLIMIT = stoi(line1);
	getline(parameterFile, line1);
	parameter.split_threshold = stoi(line1);
	getline(parameterFile, line1);
	parameter.discard_NA_regions = stoi(line1);

	// parameterFile >> parameter.regionId;
	// parameterFile >> parameter.Epsilon;
	// parameterFile >> parameter.Pi;
	// parameterFile >> parameter.lambda;
	// parameterFile >> PIXELLIMIT;
	// parameterFile >> parameter.split_threshold;
	// parameterFile >> parameter.discard_NA_regions;

	parameter.Pi_orig = parameter.Pi;
	parameterFile.close();

	// reading weights file for multiple weights on regularization term
	ifstream weightsFile(HMFInputLocation + HMFWeights);
	vector<float> weights;
	float w;
	while (weightsFile >> w){
		weights.push_back(w);
	}

	// get parent, child and prob score
	load_data();

	#ifdef DEBUG
	cout << "Start Debugging" << endl;
	cout << "row cols total: " << parameter.ROW * parameter.COLUMN << endl;
	cout << "all nodes size: " << data.allNodes.size() << endl;
	// cout << "skipped: " << skipped_nodes << endl;
	// cout << "Src dir -1: " << src_dir_m_1 << endl;
	cout << "river_nodes size: " << data.river_nodes.size() << endl;
	#endif

	auto start_3 = std::chrono::steady_clock::now();

	node_location.resize(parameter.ROW*parameter.COLUMN, -1);
	cost_map.resize(parameter.ROW*parameter.COLUMN, -1);

	int region_id = 0;
	for (int i = 0; i < data.river_nodes.size(); i++) {
		int river_id = data.river_nodes[i];

		// for large region we want to skip regions formed from outside RGB since they are too many
		// string dataset_id = parameter.regionId;
		// if (dataset_id.compare("TCLarge") == 0){
		if (parameter.discard_NA_regions == 1){
			if (data.allNodes[river_id].parentsID.size() == 0 && data.allNodes[river_id].childrenID.size() != 0 && data.allNodes[river_id].isNa == false) { // discard regions formed from outside RGB region
				node_location[river_id] = -2; 
			}
			else {
				node_location[river_id] = 1;
			}
		}
		else{
			if (data.allNodes[river_id].parentsID.size() == 0 && data.allNodes[river_id].childrenID.size() != 0) {
				node_location[river_id] = -2; 
			}
			else {
				node_location[river_id] = 1;
			}
		}

		if (node_location[river_id] == 1)
			continue;
		
		data.allRegions.emplace_back(region_id, river_id);

		que.push(river_id); // add bfs root node to conque
		data.allNodes[river_id].regionId = region_id; // assign region_id to entry node

		region_id++;		
	}

	data.total_regions = region_id;

	cout << "total_regions: " << data.total_regions << endl;

	// get all the nodes in each zone by BFS from root node
	// vector<int> bfsVisited;
	bfsVisited.resize(parameter.ROW*parameter.COLUMN, 0);
	allNodes = data.allNodes;
	bfs.resize(data.total_regions);

	auto start_bfs = std::chrono::steady_clock::now();

	#pragma omp parallel for schedule(dynamic, 1) num_threads(nThreads) // dynamic 1 is best
	for (int region_id = 0; region_id < data.total_regions; region_id++) {
		Regions &curr_region = data.allRegions[region_id];

		if (curr_region.bfsRootNode == -1) {
			curr_region.bfsOrder = {};
		}
		else {
			curr_region.bfsOrder = getBFSOrder(curr_region.bfsRootNode, bfsVisited, region_id, true);
		}

		curr_region.regionSize = curr_region.bfsOrder.size();

		// aggregate and split the tree
		#ifdef AGG_N_SPLIT
		aggNsplit(i);
		#endif
	}

	// getBFSOrderParallelV2();

	auto end_bfs = std::chrono::steady_clock::now();
	auto elapsed_bfs = end_bfs - start_bfs;
	std::cout << "BFS time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_bfs).count() << " microseconds" << endl << endl;

	#ifdef AGG_N_SPLIT
	cout << "Total split regions: " << data.splitRegionIds.size() << endl;
	#endif

	// // Calculate Cost (current elevation - entry point elevation) // added inside BFS for saving time
	// #pragma omp parallel for schedule(dynamic, batch_size) num_threads(nThreads)
	// for (int i = 0; i < data.total_regions; i++) {
	// 	Regions &curr_region = data.allRegions[i];

	// 	// for parallel BFS V2
	// 	// curr_region.bfsOrder = bfs[i];
	// 	// curr_region.regionSize = bfs[i].size();

	// 	int root_node = curr_region.bfsRootNode;
	// 	float root_elevation = data.allNodes[root_node].elevation;

	// 	for (int treeIndex = 0; treeIndex < curr_region.regionSize; treeIndex++) {

	// 		int pixelId = curr_region.bfsOrder[treeIndex];
	// 		data.allNodes[pixelId].regionId = curr_region.regionId;
	// 		float current_elevation = data.allNodes[pixelId].elevation;
	// 		data.allNodes[pixelId].cost = current_elevation - root_elevation;
	// 		cost_map[pixelId] = current_elevation - root_elevation;

	// 		if (pixelId == root_node)
	// 			node_location[pixelId] = -2;
	// 		else
	// 			node_location[pixelId] = i;
	// 	}
	// }

	#ifdef AGG_N_SPLIT
	// BFS again to form sub-regions from new roots
	bfsVisited.clear();
	bfsVisited.resize(parameter.ROW*parameter.COLUMN, 0);
	data.allRegions.clear();
	for (int sregion_id=0; sregion_id < data.splitRegionIds.size(); sregion_id++){
		int root_node = data.splitRegionIds[sregion_id];

		data.allRegions.emplace_back(sregion_id, root_node);
		Regions &curr_region = data.allRegions.back();

		curr_region.origRootId = data.rootId2OrigRootId[root_node];
		curr_region.bfsOrder = getSplitBFSOrder(curr_region.bfsRootNode, bfsVisited, sregion_id);
		curr_region.regionSize = curr_region.bfsOrder.size();
	}

	// after splitting
	data.total_regions = data.allRegions.size();

	#endif


	// // code after split regions
	// for (int sregion_id = 0; sregion_id < data.splitRegionIds.size(); sregion_id++) {
	// 	Regions curr_region = data.allRegions[sregion_id];
	// 	int root_node = data.splitRegionIds[sregion_id];

	// 	for (int p = 0; p < curr_region.regionSize; p++) {
	// 		int node_id = curr_region.bfsOrder[p];
	// 		data.allNodes[node_id].regionId = sregion_id;

	// 		if (node_id == root_node){
	// 			node_location[node_id] = -2; // indicate new root of sub-tree
	// 		}
	// 		else{
	// 			node_location[node_id] = sregion_id;
	// 		}
	// 	}
	// }

	#ifdef AGG_N_SPLIT
	for (int i=0; i < parameter.allPixelSize; i++){
		Node &curr_node = data.allNodes[i];

		// clear up previous child and parent relation
		curr_node.parentsID.clear();
		curr_node.childrenID.clear();

		// assing new parent and children
		curr_node.parentsID = curr_node.sparentsID;
		curr_node.childrenID = curr_node.schildrenID;
	}
	#endif


	// data.region_map = new int* [parameter.ROW];
	// int index = 0;
	// for (int row = 0; row < parameter.ROW; row++)
	// {
	// 	data.region_map[row] = new int[parameter.COLUMN];
	// 	for (int col = 0; col < parameter.COLUMN; col++)
	// 	{
	// 		if (node_location[index] == -2){
	// 			// use -2 for boundary nodes in the river
	// 			data.region_map[row][col] = -2;
	// 		}
	// 		else if (node_location[index] == 1){
	// 			// use 1 for nodes completely inside river
	// 			data.region_map[row][col] = 1;
	// 		}
	// 		else {
	// 			data.region_map[row][col] = node_location[index];
	// 		}
	// 		++index;
	// 	}
	// }

	// cout << "Writing region map to tiff" << endl;
	// writeRegionMap();
	// cout << "Total Large Regions: " << data.largeRegionIds.size() << endl;

	// // return;

	// // writeCostMap(cost_map);

	// // sort nodes in each region by cost
	// for (int i = 0; i < data.total_regions; i++) {
	// 	Regions &curr_region = data.allRegions[i];

	// 	#ifdef AGG_N_SPLIT
	// 	// store parent and children regions
	// 	int root_node = curr_region.bfsRootNode;
	// 	if (root_node != curr_region.origRootId){
	// 		int parent_node = data.allNodes[root_node].origParentID;
	// 		int parent_region_id = data.allNodes[parent_node].regionId;

	// 		if (parent_region_id == -1){
	// 			curr_region.parentID = -1;
	// 			continue;
	// 		}

	// 		curr_region.parentID = parent_region_id; // add parent
	// 		data.allRegions[parent_region_id].childrenID.push_back(curr_region.regionId); // add child
	// 	}
	// 	#endif
		
	// 	#ifdef HMT_Tree
	// 	// required for HMT Tree; need to sort nodes in region by elevation to from HMT Tree
	// 	curr_region.costIndexPair.resize(curr_region.regionSize);
	// 	curr_region.sortedCostIndex.resize(curr_region.regionSize);
	// 	// curr_region.index2PixelId.resize(curr_region.regionSize);

	// 	double max_cost = -INFINITY;
	// 	double min_cost = INFINITY;
	// 	for (int j = 0; j < curr_region.regionSize; j++) {
	// 		int pixelId = curr_region.bfsOrder[j];
	// 		double curr_node_cost = data.allNodes[pixelId].cost;

	// 		curr_region.index2PixelId[j] = pixelId;
	// 		curr_region.costIndexPair[j] = make_pair(curr_node_cost, j); // sort by cost

	// 		if (curr_node_cost > max_cost){
	// 			max_cost = curr_node_cost;
	// 		}

	// 		if (curr_node_cost < min_cost && curr_node_cost > 0){
	// 			min_cost = curr_node_cost;
	// 		}
				
	// 	}
	// 	sort(std::begin(curr_region.costIndexPair), std::end(curr_region.costIndexPair));

	// 	for (int k = 0; k < curr_region.regionSize; k++) {
	// 		int index = curr_region.costIndexPair[k].second;
	// 		curr_region.sortedCostIndex[index] = k; // get sorted index 'k' based on original index 'index'
	// 	}

	// 	curr_region.max_cost = max_cost;
	// 	#endif
	// }

	#ifdef HMT_Tree
	cout << "Creating HMT Tree" << endl;
	HMTTree();
	cout << "HMT Tree created" << endl;
	#endif

	auto end_3 = std::chrono::steady_clock::now();
	auto elapsed_seconds_3 = end_3 - start_3;
	std::cout << "Zone partitioning: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds_3).count() << " microseconds" << endl << endl;

	// return; // !!!!!!!!!!!!!!!!!!!!!!!!!!
	
	auto start_4 = std::chrono::steady_clock::now();

	//convert parameter Pi, M, Epsilon to log form
	parameter.Pi = eln(parameter.Pi);
	parameter.Epsilon = eln(parameter.Epsilon);

	UpdateTransProb();

	auto end_4 = std::chrono::steady_clock::now();
	auto elapsed_seconds_4 = end_4 - start_4;
	std::cout << "Handling Probability: " << std::chrono::duration_cast<std::chrono::seconds>(elapsed_seconds_4).count() << " seconds" << endl << endl;

	// inference();
	// prediction_D8_tree();

	auto start_5 = std::chrono::steady_clock::now();

	std::cout << "# threads: " << nThreads << std::endl;
	int nThreadsSort = min(nThreads, 16);
	tbb::task_scheduler_init init(nThreadsSort);
	

	// get adjacent regions and adjacent pixels
	cout << "Constructing Region Graph" << endl;

	#pragma omp parallel for schedule(dynamic, 2) num_threads(nThreads)
	for (int region_id=0; region_id < data.total_regions; region_id++){
		adjacentRegions(region_id);
	}

	// sort regions by zone size for coloring
	sort(data.regions.begin(), data.regions.end(), sortByZoneSize);
	sort(data.large_regions.begin(), data.large_regions.end(), sortByZoneSize);
	sort(data.small_regions.begin(), data.small_regions.end(), sortByZoneSize);

	cout << "Region Graph constructed" << endl;

	auto end_5 = std::chrono::steady_clock::now();
	auto elapsed_seconds_5 = end_5 - start_5;
	std::cout << "Region Graph construction time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_seconds_5).count() << " microseconds" << endl << endl;

	data.region_map = new int* [parameter.ROW];
	int index = 0;
	for (int row = 0; row < parameter.ROW; row++)
	{
		data.region_map[row] = new int[parameter.COLUMN];
		for (int col = 0; col < parameter.COLUMN; col++)
		{
			if (node_location[index] == -2){
				// use -2 for boundary nodes in the river
				data.region_map[row][col] = -2;
			}
			else if (node_location[index] == 1){
				// use 1 for nodes completely inside river
				data.region_map[row][col] = 1;
			}
			else if (data.allNodes[index].isBoundaryNode){
				data.region_map[row][col] = -20;
			}
			else {
				data.region_map[row][col] = node_location[index];
			}
			// if (data.allNodes[index].isBoundaryNode){
			// 	data.region_map[row][col] = -20;
			// }
			// else{
			// 	data.region_map[row][col] = 1;
			// }

			++index;
		}
	}

	// cout << "Writing region map to tiff" << endl;
	writeRegionMap();

	auto start_7 = std::chrono::steady_clock::now();	

	// graph coloring
	// coloring();
	coloring_v2();

	#ifdef DEBUG
	ofstream color_set_file;
	color_set_file.open(HMFOutputLocation + "color_set_stat.csv");

    cout << "Color Sets" << endl;
	int counter = 0;
    for (int i=0; i<data.color_sets.size(); i++){
		int color_set_size = data.color_sets[i].size();
		cout << "Color set " << i << " size: " << color_set_size << endl;
			
		for (int j=0; j<color_set_size; j++){
			int regionId = data.color_sets[i][j];
			int regionSize = data.allRegions[regionId].regionSize;
			color_set_file << regionId;
			if (j < color_set_size){
				color_set_file << ",";
			}
			counter++;
		}
		color_set_file << endl;
    }

	color_set_file.close();
	cout << "Colored zones counter: " << counter << endl;
	#endif

	auto end_7 = std::chrono::steady_clock::now();
	std::cout << "Coloring Time: " << std::chrono::duration_cast<std::chrono::seconds>(end_7 - start_7).count() << " seconds" << endl << endl;

	parameter.Pi = parameter.Pi_orig;
	parameter.rho = 0.999;

	// these can be one-time computation, no need to repeat again and again
	parameter.elnPi = eln_ll(parameter.Pi);
	parameter.elnPiInv = eln_ll(1 - parameter.Pi);
	parameter.elnrho = eln_ll(parameter.rho);
	parameter.elnrhoInv = eln_ll(1 - parameter.rho);
	parameter.elnHalf = eln_ll(0.5);

	auto start_6 = std::chrono::steady_clock::now();

	// calculate loglikelihood for regularization
	cout << "Getting Log Likelihood" << endl;

	#pragma omp parallel for schedule(dynamic, batch_size) num_threads(nThreads)
	for (int region_idx=0; region_idx < data.total_regions; region_idx++){
		int region_id = data.regions[region_idx].RegionId;
		getLoglikelihood(region_id);
	}

	auto end_6 = std::chrono::steady_clock::now(); 
    auto elapsed_6 = end_6 - start_6;
    std::cout << "LL Time: " << std::chrono::duration_cast<std::chrono::seconds>(elapsed_6).count() << " seconds" << std::endl;

	// see initial prediction based on max LL
	auto start_p = std::chrono::steady_clock::now(); 

	string mode = "LL";
	prediction(mode, 0.0);

	auto end_p = std::chrono::steady_clock::now();
	std::cout << "Prediction using LL Time: " << std::chrono::duration_cast<std::chrono::seconds>(end_p - start_p).count() << " seconds" << endl << endl;

	size_t violating_pairs = 0;
	violating_pairs = countViolatingPairs();

	cout << "Total Violating Pairs before Regularization: " << violating_pairs << endl;

	// validate coloring by writing to raster file
	// plot_colors();

	for (int wt=0; wt < weights.size(); wt++){
		// update frontier node based on color set
		
		// int wt = 0;
		data.prev_flood_count = 0;
		data.curr_flood_count = 0;

		data.prev_flipped_count = 0;
		data.curr_flipped_count = 0;

		cout << endl << "----------------WEIGHT: " << weights[wt] << " ------------" << endl;

		auto start_8 = std::chrono::steady_clock::now();

		updateFrontierParallel(weights[wt]);
		// updateFrontierSerial(weights[wt]);

		auto end_8 = std::chrono::steady_clock::now();
		std::cout << "New Algorithm Time: " << std::chrono::duration_cast<std::chrono::seconds>(end_8 - start_8).count() << " seconds" << endl << endl;

		auto start_pr = std::chrono::steady_clock::now(); 

		mode = "Reg";
		prediction(mode, weights[wt]);

		auto end_pr = std::chrono::steady_clock::now();
		std::cout << "Prediction using Reg Time: " << std::chrono::duration_cast<std::chrono::seconds>(end_pr - start_pr).count() << " seconds" << endl << endl;

		violating_pairs = countViolatingPairs();
		cout << "Total Violating Pairs after Regularization: " << violating_pairs << endl;

		for (int r = 0; r < data.total_regions; r++){
			// if (data.allRegions[r].regionSize >= PIXELLIMIT){
			// 	cout << "Region Id: " << data.allRegions[r].regionId << " Flood Level corr. to max LL: " << data.allRegions[r].inferredMaxCost;
			// 	cout << " Updated Flood Level: " << data.allRegions[r].frontierCost;
			// 	cout << " Max Flood Level: " << data.allRegions[r].max_cost << endl;
			// }

			// reset to initial for new weight
			double initial_cost = data.allRegions[r].inferredMaxCost;
			double initial_node = data.allRegions[r].inferredFrontierNodeIdx;
			data.allRegions[r].frontierCost = initial_cost;
			data.allRegions[r].frontierNodeIdx = initial_node;
			data.allRegions[r].MIN_LOSS = INFINITY;
		}

		cout << "--------------------------" << endl;
	}

	omp_destroy_lock(&writelock);
	
	// //Free each sub-array
    // for(int i = 0; i < parameter.ROW; ++i) {
    //     delete[] data.region_map[i];   
    // }

    // //Free the array of pointers
    // delete[] data.region_map;
}

size_t cFlood::countViolatingPairs(){
	size_t violating_pairs = 0;
	for (int rid=0; rid<data.allRegions.size(); rid++){
		Regions& curr_region = data.allRegions[rid];
		vector<AdjacencyList> neighbors = curr_region.adjacencyList;

		for (int nIdx=0; nIdx < neighbors.size(); nIdx++){
			AdjacencyList& adjacency_list = neighbors[nIdx];
			int adj_region_id = adjacency_list.regionId;
			vector<AdjacentNodePair> adj_nodes = adjacency_list.adjacentNodes; // adjacent pixel pair list sorted by nei cost as primary key

			// get adjacent region's initial flood frontier
			Regions &adj_region = data.allRegions[adj_region_id];
			double adjFloodLevel = adj_region.frontierCost;
			double currFloodLevel = curr_region.frontierCost;

			// check each adjacent pairs and see if they violate elevation-guidance
			vector<AdjacentNodePair> adjacentNodes = adjacency_list.adjacentNodes;
			for (int pIdx=0; pIdx<adjacentNodes.size(); pIdx++){
				double currNodeCost = adjacentNodes[pIdx].currNodeCost;
				double adjNodeCost = adjacentNodes[pIdx].adjNodeCost;

				
				if (currNodeCost > currFloodLevel && adjNodeCost < adjFloodLevel){ // Dry-Flood
					if (currNodeCost < adjNodeCost || abs(currNodeCost - adjNodeCost) < 0.00000001){
						// VIOLATION
						violating_pairs++;
					}
				}
				else if (currNodeCost < currFloodLevel && adjNodeCost > adjFloodLevel){ // Flood-Dry
					if (currNodeCost > adjNodeCost || abs(currNodeCost - adjNodeCost) < 0.00000001){
						// VIOLATION
						violating_pairs++;
					}
				}
			}
		}
	}

	return violating_pairs;
}



void cFlood::prediction(string mode, float weight) {
	cout << "Prediction started!" << endl;
	mappredictions.resize(parameter.allPixelSize, -1);
	mappredictions_plot.resize(parameter.allPixelSize, -1);

	// auto start_h = std::chrono::steady_clock::now();

	// ofstream flood_stat;
	// flood_stat.open(HMFOutputLocation + mode + "_flood_stat.csv");
	// flood_stat << "OrigRegionID" << "," "RegionID" << "," << "#Nodes" << "," << "#Flood" << "," << "#Dry" << "," << "allFlood" << "," << "allDry" << "," << "isEntryPoint" << "," << "rootIsNA" << endl;

	// #pragma omp parallel for schedule(dynamic, batch_size) num_threads(nThreads)
	for (int region_id = 0; region_id < data.total_regions; region_id++) {
		Regions &curr_region = data.allRegions[region_id];
		int region_size = curr_region.regionSize;

		int origRootId = curr_region.origRootId;
		int root_node = curr_region.bfsRootNode;

		int flood_count = 0;
		int dry_count = 0;
		if (region_size != 0 && curr_region.bfsRootNode != -1) {
			// if (data.allRegions[region_id].frontierCost == -1 && data.allRegions[region_id].regionSize < PIXELLIMIT) {  //not interpolated
			// 	continue;
			// }
			for (int i = 0; i < region_size; i++) {
				int node_id = curr_region.bfsOrder[i];
				Node &curr_node = data.allNodes[node_id];

				bool is_dry = false;
				
				// Refill
				// if ((curr_node.cost <= curr_region.frontierCost && curr_node.isNa == 0) || (root_node == origRootId)) { // sub-region originating from river should always be flooded in our setting
				if (curr_node.cost <= curr_region.frontierCost && curr_node.isNa == false){
					mappredictions[node_id] = 1;
					mappredictions_plot[node_id] = 1;
					data.allNodes[node_id].label = 1;
					flood_count++;
				}
				else {
					mappredictions[node_id] = 0;
					mappredictions_plot[node_id] = 0;
					data.allNodes[node_id].label = 0;
					dry_count++;
					is_dry = true;
				}

				// color boundary nodes differently
				if (curr_node.isBoundaryNode == true){
					if (is_dry){
						mappredictions_plot[node_id] = -20;
					}
					else{
						mappredictions_plot[node_id] = -10;
					}
				}
			}
		}

		#ifdef AGG_N_SPLIT
		// for cleaning by region BFS
		int dry_threshold = 0.98 * region_size;
		if (flood_count == region_size){
			curr_region.isAllFlood = true;
		}
		else if ((dry_count >= dry_threshold) && (root_node != origRootId)){ // sub-region originating from river should not be set to allDry o/w all regions will be dry
			curr_region.isAllDry = true;
			curr_region.frontierCost = 0;
		}
		else{
			curr_region.isPartialFlood = true;
		}

		if (data.allNodes[root_node].isNa)
			curr_region.isAllDry = false;

		if (curr_region.isAllDry){
			for (int j = 0; j < region_size; j++) {
				int node_id = curr_region.bfsOrder[j];

				Node &curr_node_2 = data.allNodes[node_id];

				mappredictions[node_id] = 0;
				mappredictions_plot[node_id] = 0;

				// color boundary nodes differently
				if (curr_node_2.isBoundaryNode){
					mappredictions_plot[node_id] = -10;
				}
			}
		}

		int isEntryPoint = 0;
		if (root_node == origRootId)
			isEntryPoint = 1;
		#endif

		// flood_stat << origRootId << "," << curr_region.regionId << "," << region_size << "," << flood_count << "," << dry_count << "," << curr_region.isAllFlood << "," << curr_region.isAllDry << "," << isEntryPoint << "," << data.allNodes[root_node].isNa << endl;
	}
	// flood_stat.close();

	// nodes in river
	for (int i = 0; i < data.river_nodes.size(); i++) {
		mappredictions[data.river_nodes[i]] = 1;
		mappredictions_plot[data.river_nodes[i]] = 1;
		data.allNodes[data.river_nodes[i]].label = 1;
	}
	

	// // write prediction with boundaries
	// float** prediction_plot_temp = new float* [parameter.ROW];
	// int index2 = 0;
	// for (int row = 0; row < parameter.ROW; row++)
	// {
	// 	prediction_plot_temp[row] = new float[parameter.COLUMN];
	// 	for (int col = 0; col < parameter.COLUMN; col++)
	// 	{
	// 		prediction_plot_temp[row][col] = mappredictions_plot[index2];
	// 		index2++;
	// 	}
	// }
	// GDALDataset* srcDataset22 = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	// double geotransform22[6];
	// srcDataset22->GetGeoTransform(geotransform22);
	// const OGRSpatialReference* poSRS22 = srcDataset22->GetSpatialRef();
	// GeotiffWrite finalTiff22((HMFOutputLocation + mode + "_TEMP_" + "W_" + to_string(weight) + "_" + HMFPrediction).c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform22, poSRS22);
	// finalTiff22.write(prediction_plot_temp);

	// //Free each sub-array
    // for(int i = 0; i < parameter.ROW; ++i) {
    //     delete[] prediction_plot_temp[i];   
    // }
    // //Free the array of pointers
    // delete[] prediction_plot_temp;


	#ifdef AGG_N_SPLIT
	// TODO: Uncomment Idea 4 and next loop for cleaning
	// IDea 4:  BFS through all flood or partial flood, if encounter all dry stop and correct all below them
	//get bfs order for each tree
	vector<bool> bfsVisited;
	bfsVisited.resize(data.total_regions, false);

	cout << "Cleaning using BFS through all/partial flood!" << endl;

	for (int region_id = data.total_regions-1; region_id >= 0; region_id--) {
		// int root_node = data.allRegions[region_id].bfsRootNode;

		if (bfsVisited[region_id])
			continue;

		queue<int> que;
		que.push(region_id);

		while (!que.empty()) {
			int curr_region_id = que.front();
			Regions &curr_region = data.allRegions[curr_region_id];

			bfsVisited[curr_region_id] = true;
			que.pop();

			if (curr_region.childrenID.size() == 0)
				continue;

			for (int i = 0; i < curr_region.childrenID.size(); i++) {
				int child_region_id = curr_region.childrenID[i];
				if (!bfsVisited[child_region_id]) {
					que.push(child_region_id);
				}

				Regions &child_region = data.allRegions[child_region_id];

				if (curr_region.isAllDry){
					int region_size = child_region.regionSize;

					for (int j = 0; j < region_size; j++) {
						int node_id = child_region.bfsOrder[j];

						Node &curr_node_2 = data.allNodes[node_id];

						mappredictions[node_id] = 0;
						mappredictions_plot[node_id] = 0;

						// color boundary nodes differently
						if (curr_node_2.isBoundaryNode == true){
							mappredictions_plot[node_id] = -10;
						}
					}

					// set this region to all dry after correction
					child_region.isAllDry = true;
					child_region.isAllFlood = false;
					child_region.frontierCost = 0; // TODO
				}
			}
		}
	}


	// if none of the adjacent pixels in the border are flooded, current region should be all dry
	for (int region_id = data.total_regions-1; region_id >= 0; region_id--) {
		Regions &curr_region = data.allRegions[region_id];
		vector<AdjacencyList> adjacency_list = curr_region.adjacencyList;

		int adj_flood_pixels = 0;
		for (int nei=0; nei < adjacency_list.size(); nei++){
			vector<AdjacentNodePair> adjacent_node = adjacency_list[nei].adjacentNodes;
			for (int adj=0; adj < adjacent_node.size(); adj++){
				int nei_node = adjacent_node[adj].adjacentNode;

				if (mappredictions[nei_node] == 1){
					adj_flood_pixels++;
				}
			}
		}

		if (adj_flood_pixels == 0){

			// cout << "all_dry_nei: " << curr_region.isAllDry << " " << curr_region.isAllFlood << endl;  
			curr_region.isAllFlood = false;
			curr_region.isAllDry = true;
			curr_region.frontierCost = 0;

			for (int j = 0; j < curr_region.regionSize; j++) {
				int node_id = curr_region.bfsOrder[j];

				Node &curr_node_2 = data.allNodes[node_id];

				mappredictions[node_id] = 0;
				mappredictions_plot[node_id] = 0;

				// color boundary nodes differently
				if (curr_node_2.isBoundaryNode == true){
					mappredictions_plot[node_id] = -10;
				}
			}
		}
	}
	#endif
	// comment end


	// cout << "before pixel trace" << endl;

	// // Idea 2: Pixel trace path idea: if root node is flooded but its orig parent is not then clear this region
	// // for (int region_id = 0; region_id < data.total_regions; region_id++) {
	// for (int region_id = data.total_regions-1; region_id >= 0; region_id--) {
	// 	Regions &curr_region = data.allRegions[region_id];

	// 	int root_node = curr_region.bfsRootNode;

	// 	// entry point the river
	// 	if (root_node == curr_region.origRootId)
	// 		continue; 

	// 	int orig_parent_of_root = data.allNodes[root_node].origParentID;

	// 	Node &curr_node = data.allNodes[root_node];
	// 	Node &parent_node = data.allNodes[orig_parent_of_root];

	// 	int parent_region_id = parent_node.regionId;

	// 	Regions &parent_region = data.allRegions[parent_region_id];

	// 	if (curr_node.cost <= curr_region.frontierCost && parent_node.cost > parent_region.frontierCost){
	// 		int region_size = curr_region.regionSize;

	// 		for (int j = 0; j < region_size; j++) {
	// 			int node_id = curr_region.bfsOrder[j];

	// 			Node &curr_node_2 = data.allNodes[node_id];

	// 			mappredictions[node_id] = 0;
	// 			mappredictions_plot[node_id] = 0;

	// 			// color boundary nodes differently
	// 			if (curr_node_2.isBoundaryNode == true){
	// 				mappredictions_plot[node_id] = -10;
	// 			}
	// 		}

	// 		curr_region.frontierCost = 0;
	// 	}
	// }

	float** prediction = new float* [parameter.ROW];
	int index = 0;
	for (int row = 0; row < parameter.ROW; row++)
	{
		prediction[row] = new float[parameter.COLUMN];
		for (int col = 0; col < parameter.COLUMN; col++)
		{
			prediction[row][col] = mappredictions[index];
			index++;
		}
	}
	GDALDataset* srcDataset = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	double geotransform[6];
	srcDataset->GetGeoTransform(geotransform);
	const OGRSpatialReference* poSRS = srcDataset->GetSpatialRef();
	GeotiffWrite finalTiff((HMFOutputLocation + mode + "_" + "W_" + to_string(weight) + "_" + HMFPrediction).c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform, poSRS);
	finalTiff.write(prediction);

	// write prediction with boundaries
	float** prediction_plot = new float* [parameter.ROW];
	index = 0;
	for (int row = 0; row < parameter.ROW; row++)
	{
		prediction_plot[row] = new float[parameter.COLUMN];
		for (int col = 0; col < parameter.COLUMN; col++)
		{
			prediction_plot[row][col] = mappredictions_plot[index];
			index++;
		}
	}
	GDALDataset* srcDataset2 = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	double geotransform2[6];
	srcDataset2->GetGeoTransform(geotransform2);
	const OGRSpatialReference* poSRS2 = srcDataset2->GetSpatialRef();
	GeotiffWrite finalTiff2((HMFOutputLocation + mode + "_plot_" + "W_" + to_string(weight) + "_" + HMFPrediction).c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform2, poSRS2);
	finalTiff2.write(prediction_plot);


	//Free each sub-array
    for(int i = 0; i < parameter.ROW; ++i) {
        delete[] prediction[i];   
    }
    //Free the array of pointers
    delete[] prediction;

	//Free each sub-array
    for(int i = 0; i < parameter.ROW; ++i) {
        delete[] prediction_plot[i];   
    }
    //Free the array of pointers
    delete[] prediction_plot;

	cout << "Prediction finished!" << endl;
}


void cFlood::interpolate() {
	cout << "interpolation started!" << endl;

	//for right bank.
	int current = 0;
	while (current < data.total_regions) {
		if (data.allRegions[current].inferredMaxCost == -1 && current == 0) { // only get the regions not inferred

			//find the first reach node with non -1 max cost value
			int index = -1;
			for (int j = 1; j < data.total_regions; j++) {
				if (data.allRegions[j].inferredMaxCost != -1) {
					index = j;
					break;
				}
			}
			if (index == -1) {
				break;
			}
			double value = data.allRegions[index].inferredMaxCost;
			for (int i = 0; i < index; i++) {
				data.allRegions[i].inferredMaxCost = value; // set everything to the left of 1st inferred cost 
			}
			current = index;


		}
		else if (data.allRegions[current].inferredMaxCost != -1) {
			//two cases
				//case 1: there are n points in between next reach that has cost value
				//case 2: there is no next point
			//find index of next reach node that has cost value
			int index = -1;
			int count = 0;
			double value = data.allRegions[current].inferredMaxCost;
			for (int j = current + 1; j < data.total_regions; j++) {
				count++; // get the number of intervals between 2 inferred regions
				if (data.allRegions[j].inferredMaxCost != -1) {
					index = j;
					break;
				}
			}
			if (index == -1) {// case 2
				for (int i = current + 1; i < data.total_regions; i++) {
					data.allRegions[i].inferredMaxCost = value; // set everything to the right of last inferred cost
				}
				current = data.total_regions;
				break;
			}
			else if (count == 0 && index == current + 1) { // immediate neighbor has inferred value then that should be current node
				current = index;
			}
			else {
				double interval = (data.allRegions[index].inferredMaxCost - value) / count;
				for (int i = current + 1; i < index; i++) {
					data.allRegions[i].inferredMaxCost = data.allRegions[(i - 1)].inferredMaxCost + interval;
				}
				current = index;
			}
		}

	}
	cout << "interpolation finished!" << endl;

}

int cFlood::find(struct subset subsets[], int i)
{
	// find root and make root as parent of i (path compression)
	if (subsets[i].parent != i)
		subsets[i].parent = find(subsets, subsets[i].parent);

	return subsets[i].parent;
}

// A function that does union of two sets of x and y
// (uses union by rank)
void cFlood::Union(struct subset subsets[], int x, int y)
{
	int xroot = find(subsets, x);
	int yroot = find(subsets, y);

	// Attach smaller rank tree under root of high rank tree
	// (Union by Rank)
	if (subsets[xroot].rank < subsets[yroot].rank)
		subsets[xroot].parent = yroot;
	else if (subsets[xroot].rank > subsets[yroot].rank)
		subsets[yroot].parent = xroot;

	// If ranks are same, then make one as root and increment
	// its rank by one
	else
	{
		subsets[yroot].parent = xroot;
		subsets[xroot].rank++;
	}
}


void cFlood::HMTTree() {
	// construct HMT Tree for each region (tree with multiple parent but single child)
	for (int i = 0; i < parameter.allPixelSize; i++) {
		data.allNodes[i].childrenID.clear();
		data.allNodes[i].parentsID.clear();
	}

	// construct HMTTree
	for (int i = 0; i < data.total_regions; i++) {
		int curIdx, neighborIndex;
		int row, column;

		Regions &curr_region = data.allRegions[i];
		int region_size = curr_region.regionSize;

		vector<int> highestVertex(region_size);
		subsets = (struct subset*)malloc(region_size * sizeof(struct subset)); // pre-allocate space and then parallelize
		for (size_t j = 0; j < region_size; j++) {
			subsets[j].parent = j;
			subsets[j].rank = 0;
			highestVertex[j] = j;
		}

		for (size_t l = 0; l < region_size; l++) {
			curIdx = curr_region.costIndexPair[l].second; // costIndexPairRight is sorted by cost
			row = curIdx / parameter.COLUMN;
			column = curIdx % parameter.COLUMN;

			highestVertex[curIdx] = curIdx;

			int curIdx_orig = curr_region.index2PixelId[curIdx];

			double h1 = curr_region.sortedCostIndex[curIdx]; // get index on sorted order (lower index means low cost)

			// int pixelId = curr_region.index2PixelId[curIdx];
			// int pixelId_orig = data.allNodes[pixelId].originalId;

			// check all 8 neighbors
			for (int j = max(0, row - 1); j <= min(parameter.ROW - 1, row + 1); j++) {
				for (int k = max(0, column - 1); k <= min(parameter.COLUMN - 1, column + 1); k++) {
					int neighborIndex = j * parameter.COLUMN + k;
					
					if (neighborIndex != curIdx){
						// skip neighbor from different region
						// if (data.region_map[row][column] != data.region_map[j][k]) continue;
						if (neighborIndex >= curr_region.sortedCostIndex.size()) continue;

						int neighborIndex_orig = curr_region.index2PixelId[neighborIndex];

						// int neighborIndex = data.allRegions[i].pixelidIndexPair[neighborId]; 
						double h2 = data.allRegions[i].sortedCostIndex[neighborIndex];

						if (h1 > h2) {
							int neighComponentID = find(subsets, neighborIndex);
							int currentComponetID = find(subsets, curIdx);
							if (neighComponentID == currentComponetID) {  //this means same as root2 == root1 but we don't need to find root2  //idea if they have same room they will point to same lowest vertex
								continue;
							}
							int currentHighestNodeIdx = highestVertex[neighComponentID];
							Union(subsets, curIdx, neighborIndex);

							int currentHighestNodeIdx_orig = curr_region.index2PixelId[currentHighestNodeIdx];

							data.allNodes[currentHighestNodeIdx_orig].childrenID.push_back(curIdx_orig);
							data.allNodes[curIdx_orig].parentsID.push_back(currentHighestNodeIdx_orig);

							int newComponentID = find(subsets, curIdx);
							highestVertex[newComponentID] = curIdx;
						}
					}
				}
			}
		}
	}

	// get new BFS order from split tree and then validate
	getNewBFSOrder();

	// validate tree structure
	// validateTree();
}

// void cFlood::verifyHMTTree(){
// 	cout << "verifying HMT Tree" << endl;
// 	for (int region_idx=0; region_idx < data.total_regions; region_idx++){
// 		int d8_size = data.allRegions[region_idx].regionSize;
// 		int hmt_size = data.allRegions[region_idx].bfsOrder.size();

// 		if (d8_size != hmt_size)
// 			cout << "Error on region: " << region_idx << endl;
// 	}
// }


void cFlood::getNewBFSOrder() {
	cout << "inside get new bfs order" << endl;

	//get bfs order for each tree
	vector<int> bfsVisitedNew;
	bfsVisitedNew.resize(parameter.allPixelSize, 0);

	for (int i = 0; i < data.total_regions; i++) {
		int entry_node = data.allRegions[i].bfsRootNode;
		int child_node = -1;

		for (int j = 0; j < data.allNodes[entry_node].childrenID.size(); j++) {
			child_node = data.allNodes[entry_node].childrenID[j];
		}

		while (data.allNodes[child_node].childrenID.size() != 0) {
			child_node = data.allNodes[child_node].childrenID[0]; // get the right root node
		}

		data.allRegions[i].bfsRootNode = child_node;

		data.allRegions[i].bfsOrder.clear();
		if (child_node == -1) {
			data.allRegions[i].bfsOrder = {};
		}
		else {
			data.allRegions[i].bfsOrder = getBFSOrder(child_node, bfsVisitedNew, i, false);
		}

		// verify HMT Tree 
		int d8_size = data.allRegions[i].regionSize;
		int hmt_size = data.allRegions[i].bfsOrder.size();

		if (d8_size != hmt_size){
			cout << "Error on region: " << i << endl;
			break;
		}
	}
}





void cFlood::output() {
	auto start = std::chrono::steady_clock::now();

	float** prediction = new float* [parameter.ROW];
	int index = 0;
	for (int row = 0; row < parameter.ROW; row++)
	{
		prediction[row] = new float[parameter.COLUMN];
		for (int col = 0; col < parameter.COLUMN; col++)
		{
			prediction[row][col] = mappredictions[index];
			index++;

		}
	}
	GDALDataset* srcDataset = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	double geotransform[6];
	srcDataset->GetGeoTransform(geotransform);
	const OGRSpatialReference* poSRS = srcDataset->GetSpatialRef();

	GeotiffWrite finalTiff((HMFOutputLocation + HMFPrediction).c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform, poSRS);
	finalTiff.write(prediction);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double>elapsed_seconds = end - start;

	//Free each sub-array
    for(int i = 0; i < parameter.ROW; ++i) {
        delete[] prediction[i];   
    }
    //Free the array of pointers
    delete[] prediction;

	std::cout << "Writing Prediction File took " << elapsed_seconds.count() << "seconds" << endl;
}


void cFlood::inference() {
	vector<int> inferVisited(parameter.allPixelSize, 0);

	// inferVisited.clear();
	// inferVisited.resize(parameter.allPixelSize, 0);
	for (int i = 0; i < parameter.allPixelSize; i++) {
		data.allNodes[i].correspondingNeighbour.clear();
		data.allNodes[i].correspondingNeighbourClassOne.clear();
		data.allNodes[i].correspondingNeighbourClassZero.clear();
	}

	for (int region_idx = 0; region_idx < data.allRegions.size(); region_idx++) {
		int region_size = data.allRegions[region_idx].regionSize;
		// if (region_size < PIXELLIMIT) {
		// 	continue;
		// }

		vector<int> bfsOrder = data.allRegions[region_idx].bfsOrder;

		for (int node = region_size - 1; node >= 0; node--) {
			int cur_node_id = bfsOrder[node];
			vector<int> childrenID;
			for (int c = 0; c < data.allNodes[cur_node_id].childrenID.size(); c++) {
				int child = data.allNodes[cur_node_id].childrenID[c];
				childrenID.push_back(child);
			}

			data.allNodes[cur_node_id].fi_ChildList.resize(childrenID.size() * cNum, 0);
			for (int cls = 0; cls < cNum; cls++) {
				data.allNodes[cur_node_id].fi[cls] = 0;
				data.allNodes[cur_node_id].fo[cls] = 0;
			}

			//first figure out which neighbor fmessage passes to from current node pass n.? foNode;
			//idea: In bfs traversal order leave to root, check if next the node in bfs order is parent or child of the current node (should be child or parent of the current node)
			int foNode = -1;
			bool foNode_isChild = false;
			for (int p = 0; p < data.allNodes[cur_node_id].parentsID.size(); p++) {  //check in parent list, if found respective parent node is foNode
				int pid = data.allNodes[cur_node_id].parentsID[p];
				if (!inferVisited[pid]) {
					foNode = pid;
					break;
				}
			}
			if (foNode == -1) {
				for (int c = 0; c < childrenID.size(); c++) {
					int cid = childrenID[c];
					if (!inferVisited[cid]) {
						foNode = cid;
						foNode_isChild = true;
						break;
					}
				}
			}
			data.allNodes[cur_node_id].foNode = foNode;
			data.allNodes[cur_node_id].foNode_ischild = foNode_isChild;

			// TODO: why, if we have no children, we set foNode_isChild = true?
			// Note: this condition is never true!
			if (cur_node_id == data.allRegions[region_idx].bfsRootNode && childrenID.size() == 0) { // TODO: check this
				foNode_isChild = true;
			}

			//incoming message from visited child
			if (data.allNodes[cur_node_id].childrenID.size() > 0) {

				for (int c = 0; c < childrenID.size(); c++) {
					int child_id = childrenID[c];

					if (child_id == foNode) {
						continue;
					}
					data.allNodes[cur_node_id].correspondingNeighbour.push_back(child_id);

					// saugat: multiple parents case; not triggered in our setting (always single parent)
					for (int p = 0; p < data.allNodes[child_id].parentsID.size(); p++) {
						int pid = data.allNodes[child_id].parentsID[p];
						if (pid != cur_node_id) {
							cout << "pid check condition triggered" << endl;
							data.allNodes[cur_node_id].correspondingNeighbour.push_back(pid);
						}
					}

					// saugat: multiple parents of child case; not triggered in our setting (always single parent)
					vector<int> parentOfChildExcept_currentNode; // always empty
					for (int en = 0; en < data.allNodes[child_id].parentsID.size(); en++) {
						if (data.allNodes[child_id].parentsID[en] != cur_node_id) {
							cout << "parentOfChildExcept_currentNode check condition triggered" << endl;
							parentOfChildExcept_currentNode.push_back(data.allNodes[child_id].parentsID[en]);
						}

					}
					for (int cls = 0; cls < cNum; cls++) {  //cls represents current node class
						double max = eln(0);
						vector<int> maxCorrespondingNeighbour;
						for (int c_cls = 0; c_cls < cNum; c_cls++) { //c_cls reperesnets child class label   Yc
							// TODO: why is bitcount used here?
							int max_bitCount = 1 << parentOfChildExcept_currentNode.size();
							for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent and child class label(given by c_cls)
								double productAccumulator = data.allNodes[child_id].fo[c_cls];  //product with fo(c)
								vector<int>neighbourClass;
								neighbourClass.push_back(c_cls);
								int parentClsProd = 1; //p(c), product of parent classes for child c

								// saugat: this loop is never triggered in our case
								for (int p = 0; p < parentOfChildExcept_currentNode.size(); p++) {//calculating Product(fo(p)) for all parent of current child except the current node
									cout << "parentOfChildExcept_currentNode loop triggered" << endl;
									int pid = parentOfChildExcept_currentNode[p];
									int parentClsValue = (bitCount >> p) & 1;
									parentClsProd *= parentClsValue;
									neighbourClass.push_back(parentClsValue);
									productAccumulator = elnproduct(productAccumulator, data.allNodes[pid].fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
								}
								//multiplying P(Yc|Ypc)
								parentClsProd *= cls;
								productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[c_cls][parentClsProd]);
								if (max < productAccumulator) {
									max = productAccumulator;
									maxCorrespondingNeighbour = neighbourClass;
								}
							}
						}
						data.allNodes[cur_node_id].fi_ChildList[(c * cNum + cls)] = max;
						if (cls == 0) {
							for (int t = 0; t < maxCorrespondingNeighbour.size(); t++) {
								data.allNodes[cur_node_id].correspondingNeighbourClassZero.push_back(maxCorrespondingNeighbour[t]);
							}
						}
						else {
							for (int t = 0; t < maxCorrespondingNeighbour.size(); t++) {
								data.allNodes[cur_node_id].correspondingNeighbourClassOne.push_back(maxCorrespondingNeighbour[t]);
							}
						}
					}
				}
			}

			if (foNode_isChild) {  //means the current node has all visited parents
				if (data.allNodes[cur_node_id].parentsID.size() == 0) { // saugat: entry point in river
					for (int cls = 0; cls < cNum; cls++) {
						data.allNodes[cur_node_id].fi[cls] = parameter.elnPz[cls];
					}
				}
				else {
					for (int p = 0; p < data.allNodes[cur_node_id].parentsID.size(); p++) {
						int pid = data.allNodes[cur_node_id].parentsID[p];
						data.allNodes[cur_node_id].correspondingNeighbour.push_back(pid);
					}
					for (int cls = 0; cls < cNum; cls++) {
						double max = eln(0);
						vector<int> maxNeighbourClass;
						int max_bitCount = 1 << data.allNodes[cur_node_id].parentsID.size();
						for (int bitCount = 0; bitCount < max_bitCount; bitCount++) { //summation for each parent class label
							vector<int> parentClass;
							double productAccumulator = eln(1);
							int parentClsProd = 1;
							for (int p = 0; p < data.allNodes[cur_node_id].parentsID.size(); p++) {
								int pid = data.allNodes[cur_node_id].parentsID[p];
								int parentClsValue = (bitCount >> p) & 1;
								parentClass.push_back(parentClsValue);
								parentClsProd *= parentClsValue;
								productAccumulator = elnproduct(productAccumulator, data.allNodes[pid].fo[parentClsValue]);  //calculating Product(fo(p)) for all parent of current child except the current node
							}
							productAccumulator = elnproduct(productAccumulator, parameter.elnPz_zpn[cls][parentClsProd]);
							
							if (max < productAccumulator) {
								max = productAccumulator;
								maxNeighbourClass = parentClass;
							}
						}
						
						data.allNodes[cur_node_id].fi[cls] = max;
						if (cls == 0) {
							for (int t = 0; t < maxNeighbourClass.size(); t++) {
								data.allNodes[cur_node_id].correspondingNeighbourClassZero.push_back(maxNeighbourClass[t]);
							}
						}
						else {
							for (int t = 0; t < maxNeighbourClass.size(); t++) {
								data.allNodes[cur_node_id].correspondingNeighbourClassOne.push_back(maxNeighbourClass[t]);
							}
						}
					}
				}

				//calulating fo
				for (int cls = 0; cls < cNum; cls++) { //cls represents class of the current node
					double productAccumulator = eln(1);
					for (int c = 0; c < childrenID.size(); c++) {
						int child_id = childrenID[c];
						if (child_id == foNode) {
							continue;
						}
						productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id].fi_ChildList[(c * cNum + cls)]); //multiplying with al the child fi except the outgoing child
					}
					productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id].fi[cls]);  // multiplying with fi(n)_parent

					productAccumulator = elnproduct(productAccumulator, parameter.elnPzn_xn[(cur_node_id * cNum + cls)]);
					productAccumulator = elnproduct(productAccumulator, -1 * parameter.elnPz[cls]);

					data.allNodes[cur_node_id].fo[cls] = productAccumulator;
				}

			}

			else {  //message pass n -> parent there is no fi(n)_parent   //computes for root node as well
				//calulating fo
				for (int cls = 0; cls < cNum; cls++) { //cls represents class of the current node
					double productAccumulator = eln(1);
					for (int c = 0; c < childrenID.size(); c++) {
						productAccumulator = elnproduct(productAccumulator, data.allNodes[cur_node_id].fi_ChildList[(c * cNum + cls)]); //multiplying with al the child fi except the outgoing child
					}

					productAccumulator = elnproduct(productAccumulator, parameter.elnPzn_xn[(cur_node_id * cNum + cls)]);
					productAccumulator = elnproduct(productAccumulator, -1 * parameter.elnPz[cls]);

					data.allNodes[cur_node_id].fo[cls] = productAccumulator;
				}
			}

			inferVisited[cur_node_id] = 1;
		}
	}

	updateMapPrediction();
}

void cFlood::updateMapPrediction() {
	//extracting class
	vector<int> nodeClass(parameter.allPixelSize, -1);
	mappredictions.resize(parameter.allPixelSize, -1);
	vector<int> visited(parameter.allPixelSize, 0);

	// maintain a vector of nodeId whose cost we already summed up
	vector<bool> visited_nodes(parameter.allPixelSize, false);

	cout << "updateMapPrediction" << endl;
	for (int region_idx = 0; region_idx < data.allRegions.size(); region_idx++) {
		Regions& curr_region = data.allRegions[region_idx];
		float max_cost = 0;

		// vector<float> maxcost_accumulator;
		// if (curr_region.regionSize < PIXELLIMIT) {
		// 	continue;
		// }

		// float maxCost = MAXCOST;
		int bfsroot = curr_region.bfsRootNode; //// get the root

		if (bfsroot != -1) {
			queue<int> que;
			int nodeCls;
			if (data.allNodes[bfsroot].fo[0] >= data.allNodes[bfsroot].fo[1]) {
				nodeCls = 0;
			}
			else {
				nodeCls = 1;
			}

			// set NA regions as -1
			int nodeCls_new = nodeCls;
			if (data.allNodes[bfsroot].isNa) {
				nodeCls_new = -1;
			}

			mappredictions[bfsroot] = nodeCls_new;

			nodeClass[bfsroot] = nodeCls;
			que.push(bfsroot);
			while (!que.empty()) {

				int node = que.front();
				que.pop();
				int nodeCls = nodeClass[node];
				visited[node] = 1;

				for (int c = 0; c < data.allNodes[node].correspondingNeighbour.size(); c++) {
					int neigh_id = data.allNodes[node].correspondingNeighbour[c];
					int cClass;
					if (nodeCls == 0) {	
						cClass = data.allNodes[node].correspondingNeighbourClassZero[c];
					}
					else {
						cClass = data.allNodes[node].correspondingNeighbourClassOne[c];
					}

					nodeClass[neigh_id] = cClass;

					// set NA regions as -1
					int nodeCls_new_neigh = nodeCls;
					if (data.allNodes[neigh_id].isNa) {
						nodeCls_new_neigh = -1;
					}

					mappredictions[bfsroot] = nodeCls_new_neigh;

					if (!visited[neigh_id]) {
						que.push(neigh_id);
					}

				}
			}
		}

		bool refill = false;
		if (refill){
			bool use_hmt = false;

			double SUM_COST = 0.0;
			int COUNTER = 0;
			double boundary_cost = curr_region.max_cost;


			if (use_hmt == false){
				vector<int> leafnodes;

				// get list of leaf nodes
				for (int i = 0; i < curr_region.regionSize; i++) {
					int nodeId = curr_region.bfsOrder[i];
					if (data.allNodes[nodeId].childrenID.size() == 0) {
						leafnodes.push_back(nodeId);
					}
				}

				// for D8-tree: traverse from each leaf node towards the root
				for (int l=0; l < leafnodes.size(); l++){
					int curr_node = leafnodes[l];
					while (data.allNodes[curr_node].parentsID.size() != 0) {
						// if already visited from another branch, no need to continue downwards
						if (visited_nodes[curr_node] == true)
							break;
						if (nodeClass[curr_node] == 1 && data.allNodes[curr_node].isNa == false){
							SUM_COST += data.allNodes[curr_node].cost;
							COUNTER++;
							break;
						}
						curr_node = data.allNodes[curr_node].parentsID[0];
					}
				}
			}
			else{
				int nodeId = curr_region.regionId;
				int parentId = nodeId;

				while (data.allNodes[nodeId].childrenID.size() != 0) {
					// if (data.allNodes[nodeId].cost > max_cost && data.allNodes[nodeId].isNa == false && nodeClass[nodeId] == 1) {
					// 	max_cost = data.allNodes[nodeId].cost;
					// }

					if (nodeClass[nodeId] == 0 && data.allNodes[nodeId].isNa == false) {
						if (nodeId != parentId) { // when one node is dry so all above in this branch should be dry
							SUM_COST += data.allNodes[parentId].cost;
							COUNTER++;
						}
						break;
					}
					parentId = nodeId;
					nodeId = data.allNodes[nodeId].childrenID[0];
				}

				
			}

			if (COUNTER > 0) boundary_cost = SUM_COST / COUNTER;

			// // if all the nodes are flooded for a region; get max cost among flooded
			// if (boundary_cost == -1) {
			// 	boundary_cost = max_cost;
			// }
			
			curr_region.inferredMaxCost = boundary_cost;
		}
		
	}
	cout << "Inference finished" << endl;
}

void cFlood::prediction_D8_tree() {
	cout << "Selected prediction started!" << endl;
	//mappredictions.resize(parameter.orgPixelSize, -1);

	for (int region_idx = 0; region_idx < data.allRegions.size(); region_idx++) {
		Regions curr_region = data.allRegions[region_idx];

		// if (curr_region.regionSize >= PIXELLIMIT)
		// 	cout << "inferred cost: " << curr_region.inferredMaxCost << endl;
		for (int i = 0; i < curr_region.regionSize; i++) {
			int node_id = curr_region.bfsOrder[i];
			double cost = data.allNodes[node_id].cost;

			// // refill using inferred water level
			// if (cost <= curr_region.inferredMaxCost && data.allNodes[node_id].isNa == false){
			// 	mappredictions[node_id] = region_idx;
			// }
			// else {
			// 	mappredictions[node_id] = 0;
			// }

			if (data.allNodes[node_id].isNa) {
				mappredictions[node_id] = -1;
			}
			else if (mappredictions[node_id] == 1) {
				mappredictions[node_id] = region_idx;
			}
			else {
				mappredictions[node_id] = 0;
			}
		}
	}

	for (int i = 0; i < data.river_nodes.size(); i++) {
		mappredictions[data.river_nodes[i]] = 1;
	}

	float** prediction = new float* [parameter.ROW];
	int index = 0;
	for (int row = 0; row < parameter.ROW; row++)
	{
		prediction[row] = new float[parameter.COLUMN];
		for (int col = 0; col < parameter.COLUMN; col++)
		{
			prediction[row][col] = mappredictions[index];
			index++;
		}
	}
	GDALDataset* srcDataset = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	double geotransform[6];
	srcDataset->GetGeoTransform(geotransform);
	const OGRSpatialReference* poSRS = srcDataset->GetSpatialRef();
	GeotiffWrite finalTiff((HMFOutputLocation + "_maxSum_" + HMFPrediction).c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform, poSRS);
	finalTiff.write(prediction);


	//Free each sub-array
    for(int i = 0; i < parameter.ROW; ++i) {
        delete[] prediction[i];   
    }
    //Free the array of pointers
    delete[] prediction;

	cout << "D8-based Prediction finished!" << endl;
}

void cFlood::getLoglikelihood(int region_id){
	// if (data.allRegions[region_id].regionSize >= PIXELLIMIT){
	// 	getLoglikelihoodParallel(region_id);
	// }
	// else{
	// 	getLoglikelihoodSerial(region_id);
	// }
	// getLoglikelihoodSerial(region_id);
	getLoglikelihoodParallel(region_id);
}


//As we discused, the parents cannot affect the probability of the frontier node(i.e. all parent nodes are class 1). I removed the detection of the number of parents.
// Input:	waterlikelihood: the probabilty for water node from Unet.(Not loglikelihood) The order should follow the tree sturcture.
//			drylikelihood: the probabilty for water node from Unet.The order should follow the tree sturcture.
//			treeInput: the Id of the node in tree structure. The are only nodes in main branch. The order of the nodes is from lower to higher(i.e. the first Id should be the id of the lowest node in one region, the river node).
//			Nodes: The information for parents and children.
//			treeLength: the length of the tree structure for traversing the tree structure. 
// Output:	The vector of the loglikelihood for every frontier nodes in one region.
// Global parameter: rho and Pi. In code, the names are parameter.Pi and parameter.rho. Please modify the names of these parameters
void cFlood::getLoglikelihoodSerial(int region_id) {

	double curWaterProb, curDryProb, curMaxGain = 0;
	vector<double> loglikelihood;

	Regions &curr_region = data.allRegions[region_id];
	// curr_region.nodeId2Idx.resize(curr_region.regionSize);

	// sort nodes in each region by cost in ascending order
	vector<Node>& allSortedNodes = curr_region.allSortedNodes;
	for (int j = 0; j < curr_region.regionSize; j++){
		allSortedNodes.push_back(data.allNodes[curr_region.bfsOrder[j]]);
	}
	std::sort(allSortedNodes.begin(), allSortedNodes.end(), comp);

	// computing delta for each nodes
	double initialLog = 0; // initially all nodes are dry; get sum of all dry likelihood

	auto start_4 = std::chrono::steady_clock::now(); 
	
	for (int i = 0; i < allSortedNodes.size(); i++) {
		double curWaterProb, curDryProb, curMaxGain, curNodeGain = 0;

		Node &currNode = allSortedNodes[i];
		// curr_region.nodeId2Idx[currNode.node_id] = i;

		int num_children = currNode.childrenID.size();

		data.allNodes[currNode.node_id].rNodeIdx = i;

		// set 50/50 probability for NA nodes
		if (currNode.isNa) {
			curDryProb = eln_ll(0.5);
			curWaterProb = eln_ll(0.5);
		}
		else {
			curDryProb = eln_ll(1 - currNode.p);
			curWaterProb = eln_ll(currNode.p);
		}
		
		if (currNode.parentsID.size() == 0) {
			if (currNode.childrenID.size() != 0) {
				currNode.curGain = curWaterProb - curDryProb - eln_ll(parameter.Pi) + eln_ll(1 - parameter.Pi) + num_children * eln_ll(1 - parameter.rho);
			}
			else {
				currNode.curGain = 0;
			}
		}
		else {
			if (currNode.childrenID.size() != 0) {
				currNode.curGain = curWaterProb - curDryProb + eln_ll(parameter.rho) - eln_ll(1 - parameter.rho) - eln_ll(parameter.Pi) + eln_ll(1 - parameter.Pi) + num_children * eln_ll(1 - parameter.rho);
			}
			else{
				currNode.curGain = curWaterProb - curDryProb + eln_ll(parameter.rho) - eln_ll(1 - parameter.rho) - eln_ll(parameter.Pi) + eln_ll(1 - parameter.Pi);
			}
		}

		curNodeGain = curDryProb - eln_ll(1 - parameter.Pi);
		initialLog += curNodeGain;
	}

	auto end_4 = std::chrono::steady_clock::now(); 
    auto elapsed_4 = end_4 - start_4;


	int currNodeId, curNodePosition;
	// curNode = curr_region.bfsRootNode;
	curNodePosition = 0;

	queue<int> numQue;
	curMaxGain = initialLog;

	double max_ll = -INFINITY;
	int max_node = 0;
	int frontier_node_idx = 0;

	auto start_6 = std::chrono::steady_clock::now(); 

	vector<int> container;
	container.push_back(0);

	// double prev_cost = -INFINITY;

	// NEW CODE
	if (curr_region.regionSize > 1){
		for (int idx=1; idx < curr_region.regionSize; idx++){
			Node& currNode = allSortedNodes[idx];
			currNodeId = currNode.node_id;
			double curNodeCost = currNode.cost;

			Node& prevNode = allSortedNodes[container.back()];
			int prevNodeId = prevNode.node_id;
			double prevNodeCost = prevNode.cost;

			// bool unique_elevation_node = false;
			if (curNodeCost > prevNodeCost){
				for (int i=0; i<container.size(); i++){
					curMaxGain = curMaxGain + allSortedNodes[container[i]].curGain;
				}

				container.clear();
				container.push_back(idx);
				// unique_elevation_node = true;
				// prev_cost = curNodeCost;

				curr_region.loglikelihoodUnique.push_back(curMaxGain);
				curr_region.sortedNodes.push_back(prevNode);

				// store the node with max log-likelihood during incremental calculation
				if (curMaxGain > max_ll){
					max_ll = curMaxGain;
					max_node = prevNodeId;
				frontier_node_idx = idx - 1;
				}
			}
			else{
				container.push_back(idx);
			}
		}
	}

	// process the last batch
	for (int i=0; i<container.size(); i++){
		curMaxGain = curMaxGain + allSortedNodes[container[i]].curGain;
	}

	Node& lastNode = allSortedNodes[container.back()];
	curr_region.loglikelihoodUnique.push_back(curMaxGain);
	curr_region.sortedNodes.push_back(lastNode);

	if (curMaxGain > max_ll){
		max_ll = curMaxGain;
		max_node = lastNode.node_id;
		frontier_node_idx = curr_region.regionSize - 1;
	}


	// // OLD CODE
	// for (int idx=0; idx < curr_region.regionSize; idx++){
	// 	currNodeId = allSortedNodes[idx].node_id;

	// 	if (curNodePosition < allSortedNodes.size()){
	// 		while ((allSortedNodes[curNodePosition].cost < data.allNodes[currNodeId].cost) || abs(allSortedNodes[curNodePosition].cost - data.allNodes[currNodeId].cost) < 0.0000000001){
	// 			if (!allSortedNodes[curNodePosition].visited){
	// 				allSortedNodes[curNodePosition].visited = true;
	// 				numQue.push(curNodePosition); // numQue holds all nodes with equal or lower cost
	// 				curNodePosition++;
	// 				if (curNodePosition < allSortedNodes.size())
	// 					continue;
	// 				else
	// 					break;
	// 			}
	// 		}
	// 	}
			
	// 	int unique_elevation_node = -1;
	// 	while (!numQue.empty())
	// 	{
	// 		int curNodeNum = numQue.front();
	// 		numQue.pop();

	// 		curMaxGain = curMaxGain + allSortedNodes[curNodeNum].curGain;
	// 		unique_elevation_node = curNodeNum;
	// 	}

	// 	if (unique_elevation_node != -1){
	// 		curr_region.loglikelihoodUnique.push_back(curMaxGain);
	// 		curr_region.sortedNodes.push_back(allSortedNodes[idx]);
	// 	}

	// 	// store the node with max log-likelihood during incremental calculation
	// 	if (curMaxGain > max_ll){
	// 		max_ll = curMaxGain;
	// 		max_node = currNodeId;
	// 		frontier_node_idx = idx;
	// 	}
	// }
		
	// curr_region.maxLL = max_ll;
	double initial_cost = data.allNodes[max_node].cost;
	// curr_region.frontierNodeIdx = max_node;
	curr_region.frontierNodeIdx = frontier_node_idx;
	curr_region.inferredFrontierNodeIdx = frontier_node_idx;
	curr_region.frontierCost = initial_cost;
	curr_region.inferredMaxCost = initial_cost;

	// if (curr_region.regionSize >= PIXELLIMIT){
	// 	ll_values.close();
	// 	cost_values.close();
	// }
}

void cFlood::getLoglikelihoodParallel(int region_id) {

	double curWaterProb, curDryProb, curMaxGain = 0;
	// vector<double> loglikelihood;

	Regions &curr_region = data.allRegions[region_id];
	// curr_region.nodeId2Idx.resize(curr_region.regionSize);

	// if (curr_region.regionSize == 1){
	// 	std::cout << "Debug 11" << endl;
	// }

	// sort nodes in each region by cost in ascending order
	vector<Node>& allSortedNodes = curr_region.allSortedNodes;
	for (int j = 0; j < curr_region.regionSize; j++){
		allSortedNodes.push_back(data.allNodes[curr_region.bfsOrder[j]]);
	}
	std::sort(allSortedNodes.begin(), allSortedNodes.end(), comp);

	// if (curr_region.regionSize == 1){
	// 	std::cout << "Debug 12" << endl;
	// }

	// computing delta for each nodes
	double initialLog = 0; // initially all nodes are dry; get sum of all dry likelihood

	// if (curr_region.regionSize == 1){
	// 	nThreadsIntraZoneUB = 1;
	// }

	auto start_4 = std::chrono::steady_clock::now(); 
	
	#pragma omp parallel for schedule(dynamic) reduction(+:initialLog) num_threads(nThreadsIntraZoneUB)
	for (int i = 0; i < allSortedNodes.size(); i++) {
		double curWaterProb, curDryProb, curMaxGain, curNodeGain = 0;

		Node &currNode = allSortedNodes[i];
		int num_children = currNode.childrenID.size();

		// #pragma omp critical
		// {
			// curr_region.nodeId2Idx[currNode.node_id] = i;
		// }
		data.allNodes[currNode.node_id].rNodeIdx = i;

		// set 50/50 probability for NA nodes
		if (currNode.isNa) {
			// curDryProb = eln_ll(0.5);
			// curWaterProb = eln_ll(0.5);
			curDryProb = parameter.elnHalf;
			curWaterProb = parameter.elnHalf;
		}
		else {
			curDryProb = eln_ll(1 - currNode.p);
			curWaterProb = eln_ll(currNode.p);
		}
		
		if (currNode.parentsID.size() == 0) {
			if (currNode.childrenID.size() != 0) {
				// currNode.curGain = curWaterProb - curDryProb - eln_ll(parameter.Pi) + eln_ll(1 - parameter.Pi) + eln_ll(1 - parameter.rho);
				currNode.curGain = curWaterProb - curDryProb - parameter.elnPi + parameter.elnPiInv + num_children * parameter.elnrhoInv;
			}
			else {
				currNode.curGain = 0;
			}
		}
		else {
			if (currNode.childrenID.size() != 0) {
				// currNode.curGain = curWaterProb - curDryProb + eln_ll(parameter.rho) - eln_ll(1 - parameter.rho) - eln_ll(parameter.Pi) + eln_ll(1 - parameter.Pi) + eln_ll(1 - parameter.rho);
				currNode.curGain = curWaterProb - curDryProb + parameter.elnrho - parameter.elnrhoInv - parameter.elnPi + parameter.elnPiInv + num_children * parameter.elnrhoInv;
			}
			else{
				// currNode.curGain = curWaterProb - curDryProb + eln_ll(parameter.rho) - eln_ll(1 - parameter.rho) - eln_ll(parameter.Pi) + eln_ll(1 - parameter.Pi);
				currNode.curGain = curWaterProb - curDryProb + parameter.elnrho - parameter.elnrhoInv - parameter.elnPi + parameter.elnPiInv;
			}
				
		}

		// curNodeGain = curDryProb - eln_ll(1 - parameter.Pi);
		curNodeGain = curDryProb - parameter.elnPiInv;
		initialLog += curNodeGain;
	}

	auto end_4 = std::chrono::steady_clock::now(); 
    auto elapsed_4 = end_4 - start_4;

	// if (curr_region.regionSize == 1){
	// 	std::cout << "Debug 14" << endl;
	// }

	// if (curr_region.regionSize >= PIXELLIMIT){
	// 	std::cout << "Delta Computation Time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_4).count() << " microseconds" << std::endl;
	// }


	// // initially all nodes are dry; get sum of all dry likelihood
	// double initialLog = 0;

	// #pragma omp parallel for reduction(+:initialLog)
	// for (int j = 0; j < curr_region.regionSize; j++) {
	// 	double curNodeGain, curNodeDryProb;
	// 	if (data.allNodes[curr_region.bfsOrder[j]].isNa) {
	// 		curNodeDryProb = eln_ll(0.5);
	// 	}
	// 	else {
	// 		double prob = data.allNodes[curr_region.bfsOrder[j]].p;
	// 		curNodeDryProb = eln_ll(1 - prob);
	// 	}

	// 	curNodeGain = curNodeDryProb - eln_ll(1 - parameter.Pi);
	// 	initialLog += curNodeGain;
	// }

	// auto end_5 = std::chrono::steady_clock::now(); 
    // auto elapsed_5 = end_5 - start_5;

	// if (curr_region.regionSize >= PIXELLIMIT){
	// 	std::cout << "Summation Time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_6).count() << " microseconds" << std::endl;
	// }


	int currNodeId, curNodePosition;
	// curNode = curr_region.bfsRootNode;
	curNodePosition = 0;

	queue<int> numQue;
	curMaxGain = initialLog;

	double max_ll = -INFINITY;
	int max_node = 0;
	int frontier_node_idx = 0;

	auto start_6 = std::chrono::steady_clock::now(); 

	vector<int> container;
	container.push_back(0);

	// double prev_cost = -INFINITY;

	// NEW CODE
	if (curr_region.regionSize > 1){
		for (int idx=1; idx < curr_region.regionSize; idx++){
			Node& currNode = allSortedNodes[idx];
			currNodeId = currNode.node_id;
			double curNodeCost = currNode.cost;

			Node& prevNode = allSortedNodes[container.back()];
			int prevNodeId = prevNode.node_id;
			double prevNodeCost = prevNode.cost;

			// bool unique_elevation_node = false;
			if (curNodeCost > prevNodeCost){
				for (int i=0; i<container.size(); i++){
					curMaxGain = curMaxGain + allSortedNodes[container[i]].curGain;
				}

				container.clear();
				container.push_back(idx);
				// unique_elevation_node = true;
				// prev_cost = curNodeCost;

				curr_region.loglikelihoodUnique.push_back(curMaxGain);
				curr_region.sortedNodes.push_back(prevNode);

				// store the node with max log-likelihood during incremental calculation
				if (curMaxGain > max_ll){
					max_ll = curMaxGain;
					max_node = prevNodeId;
					frontier_node_idx = idx-1;
				}
			}
			else{
				container.push_back(idx);
			}
		}
	}
	// else{
	// 	std::cout << curr_region.regionSize << endl;
	// }
	

	// process the last batch
	for (int i=0; i<container.size(); i++){
		curMaxGain = curMaxGain + allSortedNodes[container[i]].curGain;
	}

	// if (curr_region.regionSize == 1){
	// 	std::cout << "Debug 1" << endl;
	// }

	Node& lastNode = allSortedNodes[container.back()];
	curr_region.loglikelihoodUnique.push_back(curMaxGain);
	curr_region.sortedNodes.push_back(lastNode);

	// if (curr_region.regionSize == 1){
	// 	std::cout << "Debug 2" << endl;
	// }

	if (curMaxGain > max_ll){
		max_ll = curMaxGain;
		max_node = lastNode.node_id;
		frontier_node_idx = curr_region.regionSize - 1;
	}

	// if (curr_region.regionSize == 1){
	// 	std::cout << "Debug 3" << endl;
	// }


	// // OLD CODE
	// for (int idx=0; idx < curr_region.regionSize; idx++){
	// 	currNodeId = allSortedNodes[idx].node_id;

	// 	if (curNodePosition < allSortedNodes.size()){
	// 		while ((allSortedNodes[curNodePosition].cost < data.allNodes[currNodeId].cost) || abs(allSortedNodes[curNodePosition].cost - data.allNodes[currNodeId].cost) < 0.0000000001){
	// 			if (!allSortedNodes[curNodePosition].visited){
	// 				allSortedNodes[curNodePosition].visited = true;
	// 				numQue.push(curNodePosition); // numQue holds all nodes with equal or lower cost
	// 				curNodePosition++;
	// 				if (curNodePosition < allSortedNodes.size())
	// 					continue;
	// 				else
	// 					break;
	// 			}
	// 		}
	// 	}
			
	// 	int unique_elevation_node = -1;
	// 	while (!numQue.empty())
	// 	{
	// 		int curNodeNum = numQue.front();
	// 		numQue.pop();

	// 		curMaxGain = curMaxGain + allSortedNodes[curNodeNum].curGain;
	// 		unique_elevation_node = curNodeNum;
	// 	}

	// 	if (unique_elevation_node != -1){
	// 		curr_region.loglikelihoodUnique.push_back(curMaxGain);
	// 		curr_region.sortedNodes.push_back(allSortedNodes[idx]);
	// 	}

	// 	// store the node with max log-likelihood during incremental calculation
	// 	if (curMaxGain > max_ll){
	// 		max_ll = curMaxGain;
	// 		max_node = currNodeId;
	//		frontier_node_idx = idx;
	// 	}
	// }

	auto end_6 = std::chrono::steady_clock::now(); 
    auto elapsed_6 = end_6 - start_6;

	// cout << region_id << "\t" << curr_region.regionSize << "\t" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_4).count() << "\t" << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_6).count() << endl;
		
	// curr_region.maxLL = max_ll;
	double initial_cost = data.allNodes[max_node].cost;
	// curr_region.frontierNodeIdx = max_node;
	curr_region.frontierNodeIdx = frontier_node_idx;
	curr_region.inferredFrontierNodeIdx = frontier_node_idx;
	curr_region.frontierCost = initial_cost;
	curr_region.inferredMaxCost = initial_cost;

	// if (curr_region.regionSize == 1){
	// 	std::cout << "Debug 4" << endl;
	// }
}


void cFlood::adjacentRegions(int region_id){

	vector<bool> isInserted;
	isInserted.resize(data.total_regions, false);

	vector<int> insertionIndex;
	insertionIndex.resize(data.total_regions, -1);

	vector<int> adjacent_regions; // for color set

	Regions &curr_region = data.allRegions[region_id];

	
	for (int i = 0; i < curr_region.bfsOrder.size(); i++) {
		// // TODO: skip the region if its all Dry
		// if (curr_region.isAllDry)
		// 	continue;

		// // only color partially Flooded regions
		// if (curr_region.isAllDry || curr_region.isAllFlood) // TODO
		// 	continue;
		
		int current_node = curr_region.bfsOrder[i];

		int row = (int)(current_node / parameter.COLUMN);
		int col = current_node % parameter.COLUMN;
		
		for (int j=-1; j<2; j++){
			for (int k=-1; k<2; k++){
				if (j == 0 && k == 0){
					continue;
				}
				
				int row_nei = row + j;
				int col_nei = col + k;
				
				if (row_nei < 0 || col_nei < 0 || row_nei >= parameter.ROW || col_nei >= parameter.COLUMN){
					continue;
				}

				int neigh_node = row_nei * parameter.COLUMN + col_nei;

				// cout << "after neigh_node" << endl;
				
				int neighRegionId = data.allNodes[neigh_node].regionId;

				if (region_id == neighRegionId)
					continue;
				
				if (neighRegionId == -1)
					continue;

				// for visualizing region boundary as white
				data.allNodes[current_node].isBoundaryNode = true;

				// // TODO: skip the neighbor region if its all Dry
				// if (data.allRegions[neighRegionId].isAllDry) // TODO
				// 	continue;

				// // discard small regions while using Water Level difference as regularization term
				// if ((regType == 1) && (data.allRegions[neighRegionId].regionSize < PIXELLIMIT)){
				// 	continue;
				// }

				vector<AdjacencyList> &adjacency_list = curr_region.adjacencyList;
				int total_adjacent = adjacency_list.size();

				if (!isInserted[neighRegionId]){
					isInserted[neighRegionId] = true;
					insertionIndex[neighRegionId] = total_adjacent;

					adjacency_list.push_back(AdjacencyList());
					insertAdjacent(adjacency_list, current_node, neigh_node, total_adjacent, neighRegionId);

					adjacent_regions.push_back(neighRegionId);
				}
				else{
					int my_index = insertionIndex[neighRegionId];
					insertAdjacent(adjacency_list, current_node, neigh_node, my_index, neighRegionId);
				}
				// cout << "last line" << endl;

				// adjacent_regions.insert(neighRegionId);
			}
		}
	}

	#pragma omp critical
	{
		data.regions.emplace_back(region_id, curr_region.regionSize, adjacent_regions);
		if (curr_region.regionSize > 1500000){
			// data.large_regions.push_back(region_id);
			data.large_regions.emplace_back(region_id, curr_region.regionSize, adjacent_regions);
		}
		else{
			// data.small_regions.push_back(region_id);
			data.small_regions.emplace_back(region_id, curr_region.regionSize, adjacent_regions);
		}
	}
	
    // RegionVertex & region = data.adjacencyList.back();
	// region.RegionId = region_id;
	// region.regionSize = curr_region.regionSize;
    // region.neighbors = adjacent_regions;
	
	// sort adjacency list by elevation
	vector<AdjacencyList> &adjacency_list = curr_region.adjacencyList;
	for (int r = 0; r < adjacency_list.size(); r++){
		vector<AdjacentNodePair> &adjacent_nodes = adjacency_list[r].adjacentNodes;
		
		// auto starta = std::chrono::steady_clock::now(); // !!!!!!!!!!!!

		// if (adjacent_nodes.size() < 5000){
		// 	std::sort(adjacent_nodes.begin(), adjacent_nodes.end(), sortByAdjacent);
		// }
		// else{
		// 	tbb::parallel_sort(adjacent_nodes.begin(), adjacent_nodes.end(), sortByAdjacent);
		// }	

		std::sort(adjacent_nodes.begin(), adjacent_nodes.end(), sortByAdjacent);

		// tbb::parallel_sort(adjacent_nodes.begin(), adjacent_nodes.end(), sortByAdjacent);

		// auto enda = std::chrono::steady_clock::now(); // !!!!!!!!!!!!
		// std::cout << region_id << "\t" << r << "\t" << adjacent_nodes.size() << "\t" << std::chrono::duration_cast<std::chrono::nanoseconds>(enda - starta).count() << endl; // !!!!!!!!!!!!
		// std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(enda - starta).count() << endl; // !!!!!!!!!!!!
	}
}


void cFlood::coloring(){
	// // sort regions by zone size for coloring
	// sort(data.regions.begin(), data.regions.end(), sortByZoneSize);

    int maxno = 1;
	vector<unordered_map<int, RegionVertex*> > C;
    C.resize(3);

    data.color_sets.resize(3);

    for (int cur=0; cur < data.regions.size(); cur++){
        int k = 0;
        RegionVertex & p = data.regions[cur];

        int regionId = p.RegionId;
        vector<int> nbs = p.neighbors;

		Regions &curr_region = data.allRegions[regionId];

		// // regularize all Flood and partial Flood regions
		// if (curr_region.isAllDry)
		// 	continue;

		// regularize only partial Flood regions
		// if (curr_region.isAllDry || curr_region.isAllFlood) // TODO
		// 	continue;

		// skip regions with no neighbors
		if (nbs.size() == 0)
			continue;
        
        // coloring
        bool stop = false;
        while(!stop){
            int pos = 0;
            for(; pos < nbs.size(); pos++){ // go through the neighbors
				int nb = nbs[pos];
                if (C[k].find(nb) != C[k].end()) break; // find a neighbor in C[k]
            }

            if (pos == nbs.size()) stop = true;
            else k++;
        }

        if (k > maxno){
            maxno = k;
            C.resize(k + 2);
            data.color_sets.resize(k+2);
        }
        C[k][regionId] = &p;

        // data.color_sets_init[k].insert(regionId);
		data.color_sets[k].push_back(regionId);

		curr_region.color = k;
    }
}

void cFlood::coloring_v2(){

	vector<int> sorted_index;
    sorted_index.resize(data.total_regions, -1);
    for (int i=0; i < data.total_regions; i++){
        sorted_index[data.regions[i].RegionId] = i;
    }
    
    vector<RegionVertex> temp_regions;
    for (int i=0; i<data.total_regions; i++){
        temp_regions.push_back(data.regions[i]);
    }
	
	int maxno = 1;
	int color = 0;
	data.color_sets.resize(3);

    int remaining_count = temp_regions.size();
    int start_pos = 0;
    
    while(remaining_count != 0){
        vector<int> nei_regions;
        int remaining_count_tmp = 0;

        for (int cur=start_pos; cur < start_pos+remaining_count; cur++){
            RegionVertex &curr_region = temp_regions[cur];

            if (curr_region.choose == 0){ // already chosen
                continue;
            }

            if (curr_region.choose == -1){
                temp_regions.push_back(curr_region);
                remaining_count_tmp++;
            }
            
            
            for (int region_id=start_pos; region_id < start_pos+remaining_count; region_id++){
                RegionVertex &choice_region = temp_regions[region_id];
                if (choice_region.choose == 1){
					// // regularize all Flood and partial Flood regions
					// if (data.allRegions[choice_region.RegionId].isAllDry)
					// 	continue;

					// regularize only partial Flood regions
					// if (data.allRegions[choice_region.RegionId].isAllDry || data.allRegions[choice_region.RegionId].isAllFlood)
					// 	continue;

                    if (color > maxno){
                        maxno = color;
                        data.color_sets.resize(color + 2);
                    }

                    data.color_sets[color].push_back(choice_region.RegionId);
                    choice_region.choose = 0; // don't choose again
                    
                    temp_regions[region_id].color = color;
					data.allRegions[choice_region.RegionId].color = color;

					bool color_added = false;
					if (data.color_sets[color].size() >= MAX_COLOR_SET_SIZE){
                        color++;
                        
                        // 	reset removed neighbors regions to 1
                        for (int n=0; n < nei_regions.size(); n++){
                            if (temp_regions[nei_regions[n]].choose != 0)
                                temp_regions[nei_regions[n]].choose = 1;
                        }
						color_added = true;
                    }

                    // remove neighbors of choice region
                    vector<int> nbs = temp_regions[region_id].neighbors;
                    for (int nei_idx=0; nei_idx < nbs.size(); nei_idx++){
                        int nei_region_id = nbs[nei_idx];
                        RegionVertex &nei_region = temp_regions[sorted_index[nei_region_id]];
                        if (nei_region.choose == 0){ // if already chosen, no need to worry about these
                            continue;
                        }
                        nei_region.choose = -1; // don't choose nei regions for this color
                        nei_regions.push_back(sorted_index[nei_region_id]);
                    }

					if (color_added){
                        region_id = start_pos; //if new color is added, start from front again to get largest region first
                    }
                }
            }
        }
        start_pos = start_pos + remaining_count;
        remaining_count = remaining_count_tmp;
        color++; // need to start new color when there are remaining regions
        
        for (int cur=start_pos; cur < start_pos+remaining_count; cur++){
            RegionVertex &curr_region = temp_regions[cur];
            curr_region.choose = 1; // reset remaining to 1 so that they can be chosen
            sorted_index[curr_region.RegionId] = cur; // sorted index changes as remaining regions are pushed back to temp_regions
        }
    }
}




void cFlood::updateFrontierSerial(float weight){
	cout << "Updating Frontier Node in serial" << endl;
	cout << "--------------------" << endl;

	// for (int r = 0; r < data.allRegions.size(); r++){
	// 	double initial_cost = data.allRegions[r].frontierCost;
	// 	data.allRegions[r].inferredMaxCost = initial_cost;
	// }

	int no_change = 0;
	int MIN_ITER = 5;

	for (int itr=0; itr < MAX_ITERATIONS; itr++){
		data.curr_flood_count = 0;

		for (int set_idx=0; set_idx<data.color_sets.size(); set_idx++){ // unique colors
			for (int srIdx=0; srIdx < data.color_sets[set_idx].size(); srIdx++){
				int region_id = data.color_sets[set_idx][srIdx];

				updateEachFrontier(region_id, weight);
			}
		}

		cout << "itr: " << itr << " prev_flood_count: " << data.prev_flood_count << " curr_flood_count: " << data.curr_flood_count << endl;
		
		if (data.curr_flood_count == data.prev_flood_count){
			no_change++;
			if (no_change >= MIN_ITER) // if the flood count does not change for at least MIN_ITER consecutive iteration, then we break
				break;
		}
		else{
			no_change = 0;
		}

		// data.prev_flood_count = data.curr_flood_count;
		data.prev_flood_count.store(data.curr_flood_count.load());
	}
}


void cFlood::calcLambda(int region_id, int nei_idx){

	// if (regType == 0){
		int unique_elevations = data.allRegions[region_id].sortedNodes.size();

		if (unique_elevations >= UNIQUE_ELEV_THRESHOLD){
			computeRegTermParallel(region_id, nei_idx);
		}
		else{
			computeRegTermSerial(region_id, nei_idx);
		}
		// computeRegTermSerial(region_id, nei_idx);
		// computeRegTermParallel(region_id, nei_idx);
	// }
	// else{
	// 	computeRegTermWaterLevelDiff(region_id, nei_idx);
	// }				
}


void cFlood::getMinLossParallel(int region_id, float weight){
	// get adjacent region and pixels
	Regions &curr_region = data.allRegions[region_id];
	vector<AdjacencyList>& adjacency_list = curr_region.adjacencyList;

	int adjSize = adjacency_list.size();
	int unique_elev_count = curr_region.loglikelihoodUnique.size();

	int nThreadsOuter, nThreadsInner;

	if (unique_elev_count >= UNIQUE_ELEV_THRESHOLD){
		nThreadsInner = min(2, nThreadsIntraZoneUB);
		nThreadsOuter = nThreadsIntraZoneUB/nThreadsInner;
	}
	else{
		nThreadsOuter = 1;
		nThreadsInner = 1;
	}

	nThreadsOuter = 1;
	nThreadsInner = 1;

	#pragma omp parallel num_threads(nThreadsOuter)
    {   
        Regions private_min_result(-1, -1);
        private_min_result.MIN_LOSS = curr_region.MIN_LOSS;
        private_min_result.frontierCost = curr_region.frontierCost;
		private_min_result.frontierNodeIdx = curr_region.frontierNodeIdx;

		// get the overall loss for each elevations based on all neighbors
		#pragma omp for schedule(static, batch_size)
		for (int idx=0; idx < unique_elev_count; idx++){ // each unique elevations
			int currNodeId = curr_region.sortedNodes[idx].node_id;

			double loss = -curr_region.loglikelihoodUnique[idx];
			if (normalize == 1){
				loss /= curr_region.regionSize;
			}
			
			// sum lambda from all the neighbors
			int lambda = 0;

			#pragma omp parallel for reduction(+:lambda) num_threads(nThreadsInner)
			for (int nIdx = 0; nIdx < adjSize; nIdx++){
				// lambda += curr_region.adjacencyList[nIdx].lambdas[idx];
				lambda += adjacency_list[nIdx].lambdas[idx];
			}

			// // Normalize the regularization term too
			// if (adjSize > 0){
			// 	lambda /= adjSize;
			// }

			loss = loss + weight * lambda;
			if (loss < private_min_result.MIN_LOSS){
				// private_min_result.frontierNodeIdx = idx;
				// private_min_result.frontierNodeIdx = curr_region.nodeId2Idx[currNodeId];
				private_min_result.frontierNodeIdx = data.allNodes[currNodeId].rNodeIdx;
				private_min_result.frontierCost = data.allNodes[currNodeId].cost;
                private_min_result.MIN_LOSS = loss;
            }
		}

		#pragma omp critical
        {
            if (private_min_result.MIN_LOSS < curr_region.MIN_LOSS) {
				curr_region.frontierNodeIdx = private_min_result.frontierNodeIdx;
				curr_region.frontierCost = private_min_result.frontierCost;
                curr_region.MIN_LOSS = private_min_result.MIN_LOSS;
            }
        }
	}

	omp_set_lock(&writelock);
	data.curr_flood_count += curr_region.frontierNodeIdx;
	omp_unset_lock(&writelock);
}

void cFlood::getMinLoss(int region_id, float weight){
	// get adjacent region and pixels
	Regions &curr_region = data.allRegions[region_id];
	vector<AdjacencyList>& adjacency_list = curr_region.adjacencyList;

	int adjSize = adjacency_list.size();
	int unique_elev_count = curr_region.loglikelihoodUnique.size();
	
	// serial version
	// get the overall loss for each elevations based on all neighbors
	for (int idx=0; idx < unique_elev_count; idx++){ // each unique elevations
		int currNodeId = curr_region.sortedNodes[idx].node_id;

		double loss = -curr_region.loglikelihoodUnique[idx];
		if (normalize == 1){
			loss /= curr_region.regionSize;
		}
		
		// sum lambda from all the neighbors
		int lambda = 0;
		for (int nIdx = 0; nIdx < adjSize; nIdx++){
			lambda += curr_region.adjacencyList[nIdx].lambdas[idx];
		}

		// // Normalize the regularization term too
		// if (adjSize > 0){
		// 	lambda /= adjSize;
		// }

		loss = loss + weight * lambda;
		if (loss < curr_region.MIN_LOSS){
			// curr_region.frontierNodeIdx = idx;
			// curr_region.frontierNodeIdx = curr_region.nodeId2Idx[currNodeId];
			curr_region.frontierNodeIdx = data.allNodes[currNodeId].rNodeIdx;
			curr_region.frontierCost = data.allNodes[currNodeId].cost;
			curr_region.MIN_LOSS = loss;
		}
	}

	for (int i = 0; i < curr_region.regionSize; i++) {
		int node_id = curr_region.bfsOrder[i];
		Node &curr_node = data.allNodes[node_id];
		
		// Check how many node classes flip
		if (curr_node.cost <= curr_region.frontierCost && curr_node.isNa == false){
			if (data.allNodes[node_id].label == 0){
				omp_set_lock(&writelock);
				data.curr_flipped_count++;
				omp_unset_lock(&writelock);
			}
			data.allNodes[node_id].label = 1;
		}
		else {
			if (data.allNodes[node_id].label == 1){
				omp_set_lock(&writelock);
				data.curr_flipped_count++;
				omp_unset_lock(&writelock);
			}
			data.allNodes[node_id].label = 0;
		}
	}

	// omp_set_lock(&writelock);
	// data.curr_flood_count += curr_region.frontierNodeIdx;
	// omp_unset_lock(&writelock);
}

struct calcLambdaTask {
	calcLambdaTask(int region_id, int nei_idx)
		:_region_id(region_id), _nei_idx(nei_idx)
	{}
	int _region_id, _nei_idx;
};

struct getMinLossTask {
	getMinLossTask(int region_idx, float weight)
		:_region_idx(region_idx), _weight(weight)
	{}
	int _region_idx;
	float _weight;
};


void cFlood::updateFrontierParallel(float weight){
	cout << "Updating Frontier Node in parallel" << endl;
	cout << "--------------------" << endl;

	int no_change = 0;
	int MIN_ITER = 5;

	for (int itr=0; itr < MAX_ITERATIONS; itr++){
		data.curr_flipped_count = 0;
		data.curr_flood_count = 0;

		for (int set_idx=0; set_idx<data.color_sets.size(); set_idx++){ // Red, Green
			vector<calcLambdaTask> calcLambdaTasks;
			vector<getMinLossTask> getMinLossTasks;

			for (int srIdx=0; srIdx < data.color_sets[set_idx].size(); srIdx++){ // Green 
				int region_id = data.color_sets[set_idx][srIdx];
				
				// get adjacent region and pixels
				Regions &curr_region = data.allRegions[region_id];
				vector<AdjacencyList>& adjacency_list = curr_region.adjacencyList;

				int adjSize = adjacency_list.size();
				int unique_elev_count = curr_region.loglikelihoodUnique.size();
				
				// add tasks
				for (int nIdx = 0; nIdx < adjSize; nIdx++){ // Red -. All neighbors of current green node
					vector<int> &lambdas = adjacency_list[nIdx].lambdas;
					lambdas.resize(unique_elev_count, -1);

					vector<int> &to_sum = adjacency_list[nIdx].to_sum;
					to_sum.resize(unique_elev_count, 0);

					// add tasks
					calcLambdaTask tInstance = {region_id, nIdx};
					calcLambdaTasks.push_back(tInstance);
				}

				getMinLossTask lInstance = {region_id, weight};
				getMinLossTasks.push_back(lInstance);
			}

			#pragma omp parallel for schedule(dynamic, batch_size) num_threads(nThreads)
			for (int i=0; i < calcLambdaTasks.size(); i++){
				calcLambda(calcLambdaTasks[i]._region_id, calcLambdaTasks[i]._nei_idx);
				// computeRegTermSerial(calcLambdaTasks[i]._region_id, calcLambdaTasks[i]._nei_idx);
			}

			#pragma omp parallel for schedule(dynamic, batch_size) num_threads(nThreads)
			for (int i=0; i < getMinLossTasks.size(); i++){
				getMinLoss(getMinLossTasks[i]._region_idx, getMinLossTasks[i]._weight);
			}
		}
		
		// cout << "itr: " << itr << " prev_flood_count: " << data.prev_flood_count << " curr_flood_count: " << data.curr_flood_count << endl;
		
		// if (data.curr_flood_count == data.prev_flood_count){
		// 	no_change++;
		// 	if (no_change >= MIN_ITER) // if the flood count does not change for at least MIN_ITER consecutive iteration, then we break
		// 		break;
		// }
		// else{
		// 	no_change = 0;
		// }

		// // // data.prev_flood_count = data.curr_flood_count;
		// data.prev_flood_count.store(data.curr_flood_count.load());

		cout << "itr: " << itr << " prev_flipped_count: " << data.prev_flipped_count << " curr_flipped_count: " << data.curr_flipped_count << endl;
		
		if (data.curr_flipped_count == data.prev_flipped_count){
			no_change++;
			if (no_change >= MIN_ITER) // if the flood count does not change for at least MIN_ITER consecutive iteration, then we break
				break;
		}
		else{
			no_change = 0;
		}

		data.prev_flipped_count.store(data.curr_flipped_count.load());
	}
}

void cFlood::insertionSort(vector<AdjacentNodePair>& adjacentNodePairs){
    int i, j;

	int currentNode;
	int adjacentNode;
    double currNodeCost;
    double adjNodeCost;

	int size = adjacentNodePairs.size();
    for (i = 1; i < size; i++) {
		currentNode = adjacentNodePairs[i].currentNode;
		adjacentNode = adjacentNodePairs[i].adjacentNode;
        currNodeCost = adjacentNodePairs[i].currNodeCost;
        adjNodeCost = adjacentNodePairs[i].adjNodeCost;

        j = i - 1;
 
        /* Move elements of adjacentNodePairs[0..i-1], that are
           greater than currNodeCost, to one
           position ahead of their current position.
           This loop will run at most k times */
        while (j >= 0 && adjacentNodePairs[j].currNodeCost > currNodeCost) {
            adjacentNodePairs[j + 1] = adjacentNodePairs[j];
            j = j - 1;
        }
		adjacentNodePairs[j + 1].currentNode = currentNode;
		adjacentNodePairs[j + 1].adjacentNode = adjacentNode;
        adjacentNodePairs[j + 1].currNodeCost = currNodeCost;
        adjacentNodePairs[j + 1].adjNodeCost = adjNodeCost;
    }

    // for (int k=0; k<adjacentNodePairs.size(); k++){
    //     cout << adjacentNodePairs[k].currNodeCost << " " << adjacentNodePairs[k].adjNodeCost << endl;
    // }
}

// modify to return approx index if not found
int cFlood::binarySearchOnSortedNodes(vector<Node> &sorted_nodes, double targetCost) {
    int n = sorted_nodes.size();
	int left = 0;
    int right = n - 1;
	int lastIndex = -1;
    
    // while (left <= right) {
    //     int mid = left + (right - left) / 2;
        
    //     if (abs(sorted_nodes[mid].cost - targetCost) < 0.0000000001) {
	// 		left = mid + 1; // continue searching on right half for duplicate values
    //     }
    //     else if (sorted_nodes[mid].cost < targetCost) {
    //         left = mid + 1; // Target is in the right half of the array
    //     }
    //     else {
    //         right = mid - 1; // Target is in the left half of the array
    //     }
    // }

	while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (abs(sorted_nodes[mid].cost - targetCost) < 0.0000000001) {
            if (mid+1 < n && abs(sorted_nodes[mid+1].cost - targetCost) < 0.0000000001){
                    left = mid + 1; // continue searching on right half for duplicate values
                }
            else{
                return mid; // return mid value if right side is different
            }
        }
        else if (mid+1 < n && abs(sorted_nodes[mid+1].cost - targetCost) < 0.0000000001){ // also check right side
            if (mid+2 < n && abs(sorted_nodes[mid+2].cost - targetCost) < 0.0000000001){
                    left = mid + 2; // continue searching on right half for duplicate values
                }
            else{
                return mid + 1; // return mid + 1 value if right side is different
            }
        }
        else if (sorted_nodes[mid].cost < targetCost) {
            left = mid + 1; // Target is in the right half of the array
        }
        else {
            right = mid - 1; // Target is in the left half of the array
        }
    }

	return right;
}

int cFlood::binarySearchPair(const vector<AdjacentNodePair> &adj_nodes, double targetCost) {
    int n = adj_nodes.size();
	int left = 0;
    int right = n - 1;
	int lastIndex = -1;

    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (abs(adj_nodes[mid].adjNodeCost - targetCost) < 0.0000000001) {
            if (mid+1 < n && abs(adj_nodes[mid+1].adjNodeCost - targetCost) < 0.0000000001){
                    left = mid + 1; // continue searching on right half for duplicate values
                }
            else{
                return mid; // return mid value if right side is different
            }
        }
        else if (mid+1 < n && abs(adj_nodes[mid+1].adjNodeCost - targetCost) < 0.0000000001){ // also check right side
            if (mid+2 < n && abs(adj_nodes[mid+2].adjNodeCost - targetCost) < 0.0000000001){
                    left = mid + 2; // continue searching on right half for duplicate values
                }
            else{
                return mid + 1; // return mid + 1 value if right side is different
            }
        }
        else if (adj_nodes[mid].adjNodeCost < targetCost) {
            left = mid + 1; // Target is in the right half of the array
        }
        else {
            right = mid - 1; // Target is in the left half of the array
        }
    }

    return right; // -1: if target < lowest value; n-1: if target > largest value; else index of largest value less than target
}


// modify to return approx index if not found
int cFlood::binarySearch(const vector<AdjacentNodePair> &adj_nodes, double targetCost) {
    int n = adj_nodes.size();
	int left = 0;
    int right = n - 1;
	int lastIndex = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (abs(adj_nodes[mid].adjNodeCost - targetCost) < 0.0000000001) {
            // return mid; // Found the target height!
			// lastIndex = mid;
			left = mid + 1; // continue searching on right half for duplicate values
        }
        else if (adj_nodes[mid].adjNodeCost < targetCost) {
            left = mid + 1; // Target is in the right half of the array
        }
        else {
            right = mid - 1; // Target is in the left half of the array
        }
    }

	// // approximate the index if target is not found; get the largest level less than the targetCost
	// if (left >= n) {
    //     // return n - 1;
	// 	lastIndex = n - 1;
    // }
    // else if (right < 0) {
    //     // return 0;
	// 	lastIndex = -1;
    // }
    // else {
	// 	lastIndex = right; 
    // }

	return right;
}

// modify to return approx index if not found
pair<int, int> cFlood::binarySearchMulti(const vector<AdjacentNodePair> &adj_nodes, double targetCost) {
    int n = adj_nodes.size();
	int left = 0;
    int right = n - 1;
	int firstIndex = -1;
	int lastIndex = -1;

	while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (abs(adj_nodes[mid].currNodeCost - targetCost) < 0.0000000001) {
			firstIndex = mid;
			right = mid - 1;
        }
        else if (mid-1 > 0 && abs(adj_nodes[mid-1].currNodeCost - targetCost) < 0.0000000001){ // also check right side
			firstIndex = mid - 1;
			right = mid;
        }
        else if (adj_nodes[mid].currNodeCost < targetCost) {
            left = mid + 1; // Target is in the right half of the array
        }
        else {
            right = mid - 1; // Target is in the left half of the array
        }
    }

	left = 0;
    right = n - 1;

	while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (abs(adj_nodes[mid].currNodeCost - targetCost) < 0.0000000001) {
			lastIndex = mid;
			left = mid + 1;
        }
        else if (mid+1 < n && abs(adj_nodes[mid+1].currNodeCost - targetCost) < 0.0000000001){ // also check right side
			lastIndex = mid + 1;
			left = mid + 2;
        }
        else if (adj_nodes[mid].currNodeCost < targetCost) {
            left = mid + 1; // Target is in the right half of the array
        }
        else {
            right = mid - 1; // Target is in the left half of the array
        }
    }
    
    // while (left <= right) {
    //     int mid = left + (right - left) / 2;
        
    //     if (abs(adj_nodes[mid].currNodeCost - targetCost) < 0.0000000001) {
    //         // return mid; // Found the target height!
	// 		firstIndex = mid;
	// 		right = mid - 1;
    //     }
    //     else if (adj_nodes[mid].currNodeCost < targetCost) {
    //         left = mid + 1; // Target is in the right half of the array
    //     }
    //     else {
    //         right = mid - 1; // Target is in the left half of the array
    //     }
    // }

	// left = 0;
    // right = n - 1;
	// while (left <= right) {
    //     int mid = left + (right - left) / 2;
        
    //     if (abs(adj_nodes[mid].currNodeCost - targetCost) < 0.0000000001) {
    //         // return mid; // Found the target height!
	// 		lastIndex = mid;
	// 		left = mid + 1;
    //     }
    //     else if (adj_nodes[mid].currNodeCost < targetCost) {
    //         left = mid + 1; // Target is in the right half of the array
    //     }
    //     else {
    //         right = mid - 1; // Target is in the left half of the array
    //     }
    // }
	
	return make_pair(firstIndex, lastIndex);
}


void cFlood::updateEachFrontier(int region_id, float weight){
	// get adjacent region and pixels
	Regions &curr_region = data.allRegions[region_id];
	vector<AdjacencyList>& adjacency_list = curr_region.adjacencyList;

	int adjSize = adjacency_list.size();
	int unique_elev_count = curr_region.loglikelihoodUnique.size();

	for (int nIdx = 0; nIdx < adjSize; nIdx++){
		vector<int> &lambdas = adjacency_list[nIdx].lambdas;
		lambdas.resize(unique_elev_count, -1);

		vector<int> &to_sum = adjacency_list[nIdx].to_sum;
		to_sum.resize(unique_elev_count, 0);

		computeRegTermSerial(region_id, nIdx);
	} 

	// // get the overall loss for each elevations based on all neighbors
	// for (int idx=0; idx < unique_elev_count; idx++){ // each unique elevations
	// 	int currNodeId = curr_region.sortedNodes[idx].node_id;

	// 	double loss = -curr_region.loglikelihoodUnique[idx];
	// 	if (normalize == 1){
	// 		loss /= curr_region.regionSize;
	// 	}
		
	// 	// sum lambda from all the neighbors
	// 	int lambda = 0;
	// 	for (int nIdx = 0; nIdx < adjSize; nIdx++){
	// 		lambda += curr_region.adjacencyList[nIdx].lambdas[idx];
	// 	}

	// 	// // Normalize the regularization term too
	// 	// if (adjSize > 0){
	// 	// 	lambda /= adjSize;
	// 	// }

	// 	loss = loss + weight * lambda;
	// 	if (loss < curr_region.MIN_LOSS){
	// 		// curr_region.frontierNodeIdx = idx;
	// 		curr_region.frontierNodeIdx = curr_region.nodeId2Idx[currNodeId];
	//      curr_region.frontierNodeIdx = data.allNodes[currNodeId].rNodeIdx;
	// 		curr_region.frontierCost = data.allNodes[currNodeId].cost;
	// 		curr_region.MIN_LOSS = loss;
	// 	}
	// }

	// omp_set_lock(&writelock);
	// data.curr_flood_count += curr_region.frontierNodeIdx;
	// omp_unset_lock(&writelock);

	getMinLoss(region_id, weight);
}

void cFlood::computeRegTermSerial(int rIdx, int nIdx){
	Regions &curr_region = data.allRegions[rIdx];
	vector<AdjacencyList>& adjacency_list = curr_region.adjacencyList; // data structure for adjacent regions
	int adj_region_id = adjacency_list[nIdx].regionId;
	vector<AdjacentNodePair> adj_nodes = adjacency_list[nIdx].adjacentNodes; // adjacent pixel pair list sorted by nei cost as primary key

	// get adjacent region's initial flood frontier
	Regions &adj_region = data.allRegions[adj_region_id];
	double adjFloodLevel = adj_region.frontierCost;

	// binary search to find a node on adjacent region with cost equal/approximately equal to adjacent region's 
	// flood level
	// approximation is needed since flood level might not always be on adjacent pixel pairs list
	// int flood_idx = binarySearch(adj_nodes, adjFloodLevel);
	int flood_idx = binarySearchPair(adj_nodes, adjFloodLevel);

	// vector<AdjacentNodePair> adj_nodes_bottom_half;
	// for (int idx=0; idx <= flood_idx; idx++){
	// 	adj_nodes_bottom_half.emplace_back(adj_nodes[idx]);
	// }

	vector<AdjacentNodePair> adj_nodes_bottom_half = {adj_nodes.begin(), adj_nodes.begin() + flood_idx + 1};

	// sort the selected pairs by current region's elevation 
	sort(adj_nodes_bottom_half.begin(), adj_nodes_bottom_half.end(), sortByCurrent);

	// vector<AdjacentNodePair> adj_nodes_top_half;
	// for (int idx=flood_idx+1; idx <= adj_nodes.size(); idx++){
	// 	adj_nodes_top_half.emplace_back(adj_nodes[idx]);
	// }

	vector<AdjacentNodePair> adj_nodes_top_half = {adj_nodes.begin() + flood_idx + 1, adj_nodes.end()};

	// sort the selected pairs by current region's elevation 
	sort(adj_nodes_top_half.begin(), adj_nodes_top_half.end(), sortByCurrent);

	vector<Node> sortedNodes = curr_region.sortedNodes;

	// get the lowest and highest touching level on current region
	int nAdjBottom = adj_nodes_bottom_half.size();
	int nAdjTop = adj_nodes_top_half.size();

	double minLevelBot = INFINITY;
	double maxLevelBot = -INFINITY;
	double minLevelTop = INFINITY;
	double maxLevelTop = -INFINITY;

	if (nAdjBottom > 0){
		minLevelBot = adj_nodes_bottom_half[0].currNodeCost;
		maxLevelBot = adj_nodes_bottom_half[nAdjBottom - 1].currNodeCost;
	}

	if (nAdjTop > 0){
		minLevelTop = adj_nodes_top_half[0].currNodeCost;
		maxLevelTop = adj_nodes_top_half[nAdjTop - 1].currNodeCost;
	}

	double lowestAdjacentLevel = min(minLevelBot, minLevelTop);
	double highestAdjacentLevel = max(maxLevelBot, maxLevelTop);

	auto start = std::chrono::steady_clock::now(); 

	// compute lambda when current node's cost is below lowestAdjacentLevel
	int lambda_bottom = 0;
	for (int idx=0; idx < adj_nodes_bottom_half.size(); idx++){
		double curr_node_cost = adj_nodes_bottom_half[idx].currNodeCost;
		double adj_node_cost = adj_nodes_bottom_half[idx].adjNodeCost;

		if (curr_node_cost > adjFloodLevel){ // no need to check beyond this point
			break;
		}

		// we need to consider only the violating pixel pairs // example: if adjFloodLevel = 5, (3,4) is violating but (3,2) is not
		if (curr_node_cost < adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
			lambda_bottom++;
		}
	}

	auto end = std::chrono::steady_clock::now(); 
    auto elapsed = end - start;

	// if (rIdx == largest_region_id && adj_region_id == largest_nei_id)
    // 	cout << "RegionId: " << largest_region_id << " Lambda summation bottom time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;

	// if (rIdx == second_largest_region_id && adj_region_id == second_largest_nei_id)
    // 	cout << "RegionId: " << second_largest_region_id << " Lambda summation bottom time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;

	start = std::chrono::steady_clock::now(); 

	// compute lambda when current node's cost is above highestAdjacentLevel
	int lambda_top = 0;
	for (int idx=adj_nodes_top_half.size()-1; idx >= 0; idx--){
		double curr_node_cost = adj_nodes_top_half[idx].currNodeCost;
		double adj_node_cost = adj_nodes_top_half[idx].adjNodeCost;

		if (curr_node_cost < adjFloodLevel){ // no need to check below this point
			break;
		}

		// we need to consider only the violating pixel pairs // example: if adjFloodLevel = 5, (3,4) is violating but (3,2) is not
		if (curr_node_cost > adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
			lambda_top++;
		}
	}

	end = std::chrono::steady_clock::now(); 
    elapsed = end - start;

	// if (rIdx == largest_region_id && adj_region_id == largest_nei_id)
    // 	cout << "RegionId: " << largest_region_id << " Lambda summation bottom time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;

	// if (rIdx == second_largest_region_id && adj_region_id == second_largest_nei_id)
    // 	cout << "RegionId: " << second_largest_region_id << " Lambda summation bottom time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;

	int lambda_bottom_fixed = lambda_bottom;
	int lambda_top_fixed = lambda_top;
	lambda_top = 0; // for top half, we increment lambda from 0 in each level

	start = std::chrono::steady_clock::now(); 

	// computing lambda for all unique elevations of current region based on current neighbor (nIdx)
	for (int idx=0; idx < sortedNodes.size(); idx++){
		double currFloodLevel = sortedNodes[idx].cost;
		int lambda = 0;

		// if ((currFloodLevel < adjFloodLevel) || (abs(currFloodLevel - adjFloodLevel) < 0.00000001)){
		// 	lambda = getLambda(adj_nodes_bottom_half, currFloodLevel, adjFloodLevel, lambda_bottom, 0);
		// }
		// // for elevations in top half
		// else if (currFloodLevel > adjFloodLevel){
		// 	lambda = getLambda(adj_nodes_top_half, currFloodLevel, adjFloodLevel, lambda_top, 1);
		// }

		// Case 1: current node's cost is equal to adjacent regions' frontier cost
		if (abs(currFloodLevel - adjFloodLevel) < 0.00000001){
			lambda = 0;
		}
		// Case 2: current node's cost is below lowestAdjacentLevel
		else if ((currFloodLevel < lowestAdjacentLevel) || (abs(currFloodLevel - lowestAdjacentLevel) < 0.00000001)){
			lambda = lambda_bottom_fixed; // all violating pairs between current region's lowest touching node 
														// and neighbors flood level 
		}
		// Case 3: current node's cost is above highestAdjacentLevel
		else if ((currFloodLevel > highestAdjacentLevel) || (abs(currFloodLevel - highestAdjacentLevel) < 0.00000001)){
			lambda = lambda_top_fixed; // all violating pairs between current region's highest touching node 
													   // and neighbors flood level 
		}
		// Case 4: current node's cost is between touching nodes' costs
		else{
			// for elevations in bottom half
			if ((currFloodLevel < adjFloodLevel) || (abs(currFloodLevel - adjFloodLevel) < 0.00000001)){
				lambda = getLambda(adj_nodes_bottom_half, currFloodLevel, adjFloodLevel, lambda_bottom, 0);
			}
			// for elevations in top half
			else if (currFloodLevel > adjFloodLevel){
				lambda = getLambda(adj_nodes_top_half, currFloodLevel, adjFloodLevel, lambda_top, 1);
			}
		}

		// save lambda for each elevation in an array
		adjacency_list[nIdx].lambdas[idx] = lambda;
	}

	end = std::chrono::steady_clock::now(); 
    elapsed = end - start;

	// if (rIdx == largest_region_id && adj_region_id == largest_nei_id)
    // 	cout << "RegionId: " << largest_region_id << " Lambda computation unique elevations time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;

	// if (rIdx == second_largest_region_id && adj_region_id == second_largest_nei_id)
    // 	cout << "RegionId: " << second_largest_region_id << " Lambda computation unique elevations time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << " microseconds" << std::endl;
}

void cFlood::computeRegTermParallel(int rIdx, int nIdx){ // curr_region, nei_region
	Regions &curr_region = data.allRegions[rIdx];
	// vector<AdjacencyList>& adjacency_list = curr_region.adjacencyList; // data structure for adjacent regions
	AdjacencyList& adjacency_list = curr_region.adjacencyList[nIdx];
	int adj_region_id = adjacency_list.regionId;
	vector<AdjacentNodePair> adj_nodes = adjacency_list.adjacentNodes; // adjacent pixel pair list sorted by nei cost as primary key

	// get adjacent region's initial flood frontier
	Regions &adj_region = data.allRegions[adj_region_id];
	double adjFloodLevel = adj_region.frontierCost;

	// binary search to find a node on adjacent region with cost equal/approximately equal to adjacent region's flood level
	// approximation is needed since flood level might not always be on adjacent pixel pairs list
	int flood_idx = binarySearchPair(adj_nodes, adjFloodLevel);

	vector<AdjacentNodePair> adj_nodes_bottom_half = {adj_nodes.begin(), adj_nodes.begin() + flood_idx + 1};

	// sort the selected pairs by current region's elevation
	sort(adj_nodes_bottom_half.begin(), adj_nodes_bottom_half.end(), sortByCurrent);

	vector<AdjacentNodePair> adj_nodes_top_half = {adj_nodes.begin() + flood_idx + 1, adj_nodes.end()};

	// sort the selected pairs by current region's elevation
	sort(adj_nodes_top_half.begin(), adj_nodes_top_half.end(), sortByCurrent);

	vector<Node> sortedNodes = curr_region.sortedNodes;

	// get the lowest and highest touching level on current region
	int nAdjBottom = adj_nodes_bottom_half.size();
	int nAdjTop = adj_nodes_top_half.size();

	double minLevelBot = INFINITY;
	double maxLevelBot = -INFINITY;
	double minLevelTop = INFINITY;
	double maxLevelTop = -INFINITY;

	if (nAdjBottom > 0){
		minLevelBot = adj_nodes_bottom_half[0].currNodeCost;
		maxLevelBot = adj_nodes_bottom_half[nAdjBottom - 1].currNodeCost;
	}

	if (nAdjTop > 0){
		minLevelTop = adj_nodes_top_half[0].currNodeCost;
		maxLevelTop = adj_nodes_top_half[nAdjTop - 1].currNodeCost;
	}

	double lowestAdjacentLevel = min(minLevelBot, minLevelTop);
	double highestAdjacentLevel = max(maxLevelBot, maxLevelTop);

	int lambda_bottom = 0;
	int lambda_top = 0;

	int total_adjacent_pairs = adj_nodes.size();

	double elapsed_1, elapsed_2, elapsed_3, elapsed_4, elapsed_5, elapsed_6;  // !!!!!!!!!!!!

	int nThreadsInner = max(1, nThreadsIntraZoneUB/2);

	int nThreadsParSection = min(2, nThreadsIntraZoneUB);
	omp_set_num_threads(nThreadsParSection);

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			int nThreadsFor = 1;
			if (adj_nodes_bottom_half.size() >= ADJ_PAIRS_THRESHOLD*25){ // if small boundary, run with single thread
				nThreadsFor = min(4, nThreadsInner); // beyond 4 is not good
			}

			#pragma omp parallel for reduction(+:lambda_bottom) num_threads(nThreadsFor)
			for (int idx=0; idx < adj_nodes_bottom_half.size(); idx++){
				double curr_node_cost = adj_nodes_bottom_half[idx].currNodeCost;
				double adj_node_cost = adj_nodes_bottom_half[idx].adjNodeCost;

				// we need to consider only the violating pixel pairs // example: if adjFloodLevel = 5, (3,4) is violating but (3,2) is not
				if (curr_node_cost < adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
					lambda_bottom++;
				}
			}
		}

		#pragma omp section
		{
			int nThreadsFor = 1;
			if (adj_nodes_top_half.size() >= ADJ_PAIRS_THRESHOLD*25){ // if small boundary, run with single thread
				nThreadsFor = min(4, nThreadsInner); // beyond 4 is not good
			}

			#pragma omp parallel for reduction(+:lambda_top) num_threads(nThreadsFor)
			for (int idx=adj_nodes_top_half.size()-1; idx >= 0; idx--){
				double curr_node_cost = adj_nodes_top_half[idx].currNodeCost;
				double adj_node_cost = adj_nodes_top_half[idx].adjNodeCost;

				// we need to consider only the violating pixel pairs // example: if adjFloodLevel = 5, (3,4) is violating but (3,2) is not
				if (curr_node_cost > adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
					lambda_top++;
				}
			}
		}
	}


	int lambda_bottom_fixed = lambda_bottom;
	int lambda_top_fixed = lambda_top;
	lambda_top = 0; // for top half, we increment lambda from 0 in each level

	// find middle index on sortedNodes by searching adjFloodlevel and divide sortedNodes into 2 halves
	int midpoint = binarySearchOnSortedNodes(sortedNodes, adjFloodLevel);

	omp_set_num_threads(nThreadsParSection);
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			int nThreadsFor = 1;
			if (adj_nodes_bottom_half.size() >= ADJ_PAIRS_THRESHOLD){ // if small boundary, run with single thread
				nThreadsFor = min(nThreadsInner, 16); // beyond 16 is not good
			}

			// calculate how much should be subtracted for each elevation for bottom half
			#pragma omp parallel for schedule(static, batch_size) num_threads(nThreadsFor) // TODO: use dynamic scheduling
			for (int idx=0; idx <= midpoint; idx++){
				double currFloodLevel = sortedNodes[idx].cost;
				adjacency_list.to_sum[idx] = getLambdaBot(adj_nodes_bottom_half, currFloodLevel);
			}

			serialPrefixSum(adjacency_list.to_sum, adjacency_list.lambdas, 0, midpoint+1, lambda_bottom); // for bottom
		}

		#pragma omp section
		{
			int nThreadsFor = 1;
			if (adj_nodes_top_half.size() >= ADJ_PAIRS_THRESHOLD){ // if small boundary, run with single thread
				nThreadsFor = min(nThreadsInner, 16); // beyond 16 is not good
			}

			// calculate how much should be added for each elevation for top half
			#pragma omp parallel for schedule(static, batch_size) num_threads(nThreadsFor)
			for (int idx=midpoint+1; idx < sortedNodes.size(); idx++){
				double currFloodLevel = sortedNodes[idx].cost;
				adjacency_list.to_sum[idx] = getLambdaTop(adj_nodes_top_half, currFloodLevel);
			}

			serialPrefixSum(adjacency_list.to_sum, adjacency_list.lambdas, midpoint+1, sortedNodes.size(), lambda_top); // for top // !!!!!!!
		}
	}		
}


// Serial prefix sum
void cFlood::serialPrefixSum(vector<int>& input, vector<int>& output, int start, int end, int lambda_init) {
    const int n = input.size();
    // vector<int> temp(n);

    int local_sum = 0;
    for (int i = start; i < end; i++) {
        local_sum += input[i];
        // temp[i] = local_sum;
		output[i] = local_sum + lambda_init;
    }

    // // Copy the result to output array
    // for (int j = start; j < end; j++) {
	// 	output[j] = temp[j] + lambda_init;	
    // }
}

class Body {
    int sum;
    vector<int> input;
    vector<int>& output;
    int lambda_init;

	public:
		Body(vector<int> input_, vector<int>& output_, int lambda_init_ ) : sum(0), input(input_), output(output_), lambda_init(lambda_init_) {}

		template<typename Tag>
		void operator()( const tbb::blocked_range<int>& r, Tag ) { // overloaded function; defines work done by each thread when processing a subrange of input (blocked range)
			int temp = sum;                                        // invoked by tbb for each subrange of elements to be processed in parallel
			for( int i=r.begin(); i<r.end(); ++i ) {
				temp = temp + input[i]; // sum of subrange is accumulated in temp
				if( Tag::is_final_scan() ) // if the scan is final, update output vector
					output[i] = temp + lambda_init;
			}
			sum = temp;

		}
    	Body( Body& b, tbb::split ) : input(b.input), output(b.output), sum(0), lambda_init(b.lambda_init) {} // create split instance when tbb divides work among threads
		void reverse_join( Body& a ) { sum = a.sum + sum; } // combine results from 2 halves of split
		void assign( Body& b ) { sum = b.sum; } // copy the sum from one body instance to another			
};

void cFlood::parallelScan(vector<int> input, vector<int>& output, int start, int end, int lambda_init) {
    Body body(input, output, lambda_init);
    tbb::parallel_scan(tbb::blocked_range<int>(start, end), body);
}

void cFlood::parallelPrefixSum(vector<int>& input, vector<int>& output, int start, int end, int lambda_init){
    
    int nThreads;
    vector<int> temp;

	// omp_set_num_threads(nThreadsIntraZoneUB);

    #pragma omp parallel 
	{
        int i;
        #pragma omp single
        {
            nThreads = omp_get_num_threads();
            // cout << "nThreads: " << nThreads << endl;
            temp.resize(nThreads, 0);
            temp[0] = 0;
        }   

        int tid = omp_get_thread_num();
        int sum = 0;

        #pragma omp for schedule(static)
        for (i=start; i < end; i++){
            sum += input[i];
            output[i] = sum;
        }

        temp[tid + 1] = sum;

        #pragma omp barrier
        int offset = 0;
        for (i=0; i<(tid+1); i++){
            offset += temp[i];
        }

        #pragma omp for schedule(static)
        for(i=start; i<end; i++){
            output[i] += lambda_init + offset;
        }
    }
    temp.clear();
}

void cFlood::computeRegTermWaterLevelDiff(int rIdx, int nIdx){
	Regions &curr_region = data.allRegions[rIdx];
	vector<AdjacencyList>& adjacency_list = curr_region.adjacencyList;
	int adj_region_id = adjacency_list[nIdx].regionId;

	// get adjacent region's initial flood frontier
	Regions &adj_region = data.allRegions[adj_region_id];
	double adjFloodLevel = adj_region.frontierCost;
	double adjMaxLevel = adj_region.max_cost;

	vector<Node> sortedNodes = curr_region.sortedNodes;

	// computing lambda for all unique elevations of current region based on current neighbor (nIdx)
	for (int idx=0; idx < sortedNodes.size(); idx++){
		double currFloodLevel = sortedNodes[idx].cost;

		int lambda = 0.0;
		lambda = abs(currFloodLevel - adjFloodLevel);

		// save lambda for each elevation in an array
		adjacency_list[nIdx].lambdas[idx] = lambda;
	}
}

// get lambda based on number of pixel pairs violating physical law within delta-h of two regions
int cFlood::getLambda(vector<AdjacentNodePair> adj_nodes, double currFloodLevel, double adjFloodLevel, int& lambda, int side){
	// search the 2 index of currFloodLevel on adj_nodes since we only update up to this level
	pair<int, int> indices = binarySearchMulti(adj_nodes, currFloodLevel);

	int lowerIdx = indices.first;
	int higherIdx = indices.second;

	// if the elevation is not found on adjacent pixel pair list, no need to update lambda
	if (lowerIdx == -1 && higherIdx == -1)
		return lambda;

	// bottom half
	if (side == 0){
		for (int idx = lowerIdx; idx <= higherIdx; idx++){
			double curr_node_cost = adj_nodes[idx].currNodeCost;
			double adj_node_cost = adj_nodes[idx].adjNodeCost;

			// we need to consider only the violating pixel pairs // example: if adjFloodLevel = 5, (3,4) is violating but (3,2) is not
			// now, if water is at level 3, then (3,4) is not violating so decrement lambda
			if (curr_node_cost < adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
				lambda--;
			}
		}
	}
	// top half
	else{
		for (int idx = lowerIdx; idx <= higherIdx; idx++){
			double curr_node_cost = adj_nodes[idx].currNodeCost;
			double adj_node_cost = adj_nodes[idx].adjNodeCost;

			// when water level rises on current region from adjFloodLevel say (5), lambda = all pairs that violates until current level
			if (curr_node_cost > adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
				lambda++;
			}		
		}
	}
	return lambda;
}

// get lambda based on number of pixel pairs violating physical law within delta-h of two regions
int cFlood::getLambdaBotDebug(vector<AdjacentNodePair> adj_nodes, double currFloodLevel){
	// search the 2 index of currFloodLevel (handle duplicate case, on current region multiple nodes can have same height) 
	// on adj_nodes since we only update up to this level
	pair<int, int> indices = binarySearchMulti(adj_nodes, currFloodLevel);

	int lowerIdx = indices.first;
	int higherIdx = indices.second;

	// if the elevation is not found on adjacent pixel pair list, no need to update lambda
	if (lowerIdx == -1 && higherIdx == -1){
		cout << currFloodLevel << "\t" << higherIdx - lowerIdx << endl;
		return 0;
	}
		

	int to_sum = 0;
	for (int idx = lowerIdx; idx <= higherIdx; idx++){
		double curr_node_cost = adj_nodes[idx].currNodeCost;
		double adj_node_cost = adj_nodes[idx].adjNodeCost;

		// we need to consider only the violating pixel pairs
		// now, if water is at level 3, then (3,4) is not violating so decrement lambda
		if (curr_node_cost < adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
			to_sum++; 
		}
	}

	cout << currFloodLevel << "\t" << higherIdx - lowerIdx << endl;

	return -to_sum;
}


// get lambda based on number of pixel pairs violating physical law within delta-h of two regions
int cFlood::getLambdaBot(vector<AdjacentNodePair> adj_nodes, double currFloodLevel){
	// search the 2 index of currFloodLevel (handle duplicate case, on current region multiple nodes can have same height) 
	// on adj_nodes since we only update up to this level
	pair<int, int> indices = binarySearchMulti(adj_nodes, currFloodLevel);

	int lowerIdx = indices.first;
	int higherIdx = indices.second;

	// if the elevation is not found on adjacent pixel pair list, no need to update lambda
	if (lowerIdx == -1 && higherIdx == -1) // water level is below the lowest touching level
		return 0;

	int to_sum = 0;
	for (int idx = lowerIdx; idx <= higherIdx; idx++){
		double curr_node_cost = adj_nodes[idx].currNodeCost;
		double adj_node_cost = adj_nodes[idx].adjNodeCost;

		// we need to consider only the violating pixel pairs
		// now, if water is at level 3, then (3,4) is not violating so decrement lambda
		if (curr_node_cost < adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
			to_sum++; 
		}
	}

	return -to_sum;
}

// // get lambda based on number of pixel pairs violating physical law within delta-h of two regions
// int cFlood::getLambdaTop(vector<AdjacentNodePair> adj_nodes, double currFloodLevel){
// 	// search the 2 index of currFloodLevel (handle duplicate case, on current region multiple nodes can have same height) 
// 	// on adj_nodes since we only update up to this level
// 	pair<int, int> indices = binarySearchMulti(adj_nodes, currFloodLevel);

// 	int lowerIdx = indices.first;
// 	int higherIdx = indices.second;

// 	// if the elevation is not found on adjacent pixel pair list, no need to update lambda
// 	if (lowerIdx == -1 && higherIdx == -1) // water level is below the lowest touching level
// 		return 0;

// 	int to_sum = 0;
// 	for (int idx = lowerIdx; idx <= higherIdx; idx++){
// 		double curr_node_cost = adj_nodes[idx].currNodeCost;
// 		double adj_node_cost = adj_nodes[idx].adjNodeCost;

// 		// when water level rises on current region from adjFloodLevel say (5), lambda = all pairs that violates until current level
// 		if (curr_node_cost > adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
// 			to_sum++;
// 		}		
// 	}

// 	return to_sum;
// }

// get lambda based on number of pixel pairs violating physical law within delta-h of two regions
int cFlood::getLambdaTop(vector<AdjacentNodePair> adj_nodes, double currFloodLevel){
	// search the 2 index of currFloodLevel (handle duplicate case, on current region multiple nodes can have same height) 
	// on adj_nodes since we only update up to this level
	pair<int, int> indices = binarySearchMulti(adj_nodes, currFloodLevel);

	int lowerIdx = indices.first;
	int higherIdx = indices.second;

	// if the elevation is not found on adjacent pixel pair list, no need to update lambda
	if (lowerIdx == -1 && higherIdx == -1) // water level is above the highest touching level
		return 0;

	int to_sum = 0;
	for (int idx = lowerIdx; idx <= higherIdx; idx++){
		double curr_node_cost = adj_nodes[idx].currNodeCost;
		double adj_node_cost = adj_nodes[idx].adjNodeCost;

		// when water level rises on current region from adjFloodLevel say (5), lambda = all pairs that violates until current level
		if (curr_node_cost > adj_node_cost || abs(curr_node_cost - adj_node_cost) < 0.00000001){
			to_sum++;
		}		
	}

	return to_sum;
	// return -to_sum;
}


void cFlood::writeRegionMap(){
	GDALDataset* srcDataset_ = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	double geotransform_[6];
	srcDataset_->GetGeoTransform(geotransform_);
	const OGRSpatialReference* poSRS_ = srcDataset_->GetSpatialRef();

	GeotiffWrite mapTiff((HMFOutputLocation + "TC" + to_string(parameter.regionId) + "_RegionMap" + ".tif").c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform_, poSRS_);
	mapTiff.writei(data.region_map);
}

void cFlood::writeCostMap(vector<float> cost_map){
	float** cost_data = new float* [parameter.ROW];
	int indexx = 0;
	for (int row = 0; row < parameter.ROW; row++)
	{
		cost_data[row] = new float[parameter.COLUMN];
		for (int col = 0; col < parameter.COLUMN; col++)
		{
			cost_data[row][col] = cost_map[indexx];
			indexx++;
		}
	}

	cout << "Writing cost map to tiff!!!" << endl;
	// save cost as a new tif file
	GDALDataset* srcDataset_3 = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	double geotransform_3[6];
	srcDataset_3->GetGeoTransform(geotransform_3);
	const OGRSpatialReference* poSRS_3 = srcDataset_3->GetSpatialRef();

	GeotiffWrite mapTiff3((HMFOutputLocation + "TC" + to_string(parameter.regionId) + "_CostMap" + ".tif").c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform_3, poSRS_3);
	mapTiff3.write(cost_data);

	//Free each sub-array
    for(int i = 0; i < parameter.ROW; ++i) {
        delete[] cost_data[i];   
    }
    //Free the array of pointers
    delete[] cost_data;
	cout << "Writing cost map to tiff completed!!!" << endl;
}

void cFlood::plot_colors(){
	cout << "Plotting Color Map" << endl;
	float** color_map;
	color_map = new float* [parameter.ROW];
	
	int node_id = 0;
	for (int row = 0; row < parameter.ROW; row++)
	{
		color_map[row] = new float[parameter.COLUMN];
		for (int col = 0; col < parameter.COLUMN; col++)
		{
			int region_id = data.allNodes[node_id].regionId;
			int color = data.allRegions[region_id].color;

			// if (data.region_map[row][col] = -2;){
			// 	// use -2 for boundary nodes in middle of river
			// 	color_map[row][col] = -2;
			// }
			if (data.region_map[row][col] == 1){
				// use -1 for nodes in river
				color_map[row][col] = -1;
			}
			else if (data.region_map[row][col] == -1){
				// use -2 for nodes not belonging to any zones
				color_map[row][col] = -2;
			}
			else {
				color_map[row][col] = color;
			}
			
			node_id++;
		}
	}

	GDALDataset* srcDataset_c = (GDALDataset*)GDALOpen((HMFInputLocation + HMFFel).c_str(), GA_ReadOnly);
	double geotransform_c[6];
	srcDataset_c->GetGeoTransform(geotransform_c);
	const OGRSpatialReference* poSRS_c = srcDataset_c->GetSpatialRef();

	GeotiffWrite mapTiff((HMFOutputLocation + "TC" + to_string(parameter.regionId) + "_ColorMap" + ".tif").c_str(), parameter.ROW, parameter.COLUMN, 1, geotransform_c, poSRS_c);
	mapTiff.write(color_map);

	//Free each sub-array
    for(int i = 0; i < parameter.ROW; ++i) {
        delete[] color_map[i];   
    }
    //Free the array of pointers
    delete[] color_map;
}


int main(int argc, char* argv[]) {
	cFlood flood;
	flood.input(argc, argv);
	return 0;
}