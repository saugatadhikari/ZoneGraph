#pragma once
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <ctime>
#include <cmath>
#include <limits>
#include <set>
#include <cstdio>
#include <unordered_map>
#include <atomic>

using namespace std;


struct RegionVertex{
    int RegionId;
	int regionSize;
    vector<int> neighbors;
	int choose = 1; // 1: choose, 0: already chosen, -1: not sure
	int color;

	RegionVertex(int _RegionId, int _regionSize, vector<int> _neighbors)
		: RegionId(_RegionId), regionSize(_regionSize), neighbors(_neighbors) {}
};

struct Node {
	int node_id;
	vector<int> parentsID; // all parents (lower elevation)
	vector<int> childrenID; // all children (higher elevation)

	double cost; // (current node elevation - elevation of entry point in river)
	float p; // flood probability from U-Net
	bool isNa = false;
	double elevation; // field elevation layer (after pit filling)

	// log likelihood calculation
	double curGain; // flood likelihood - dry likelihood
	bool visited = false; // used to store visited node

	int regionId = -1;
	int rNodeIdx = -1; // index of node in its particular region

	// store inferred class
	int label = 0;

	// // datatypes for split regions
	int stNodes = 1; // each node stores the number of sub-tree nodes
	vector<int> sparentsID;
	vector<int> schildrenID;
	int origParentID; // root node of a sub-tree points to its original parent in global-tree

	// for visualizing region boundary as white
	bool isBoundaryNode = false;

	// for inference algorithm
	vector<int> correspondingNeighbour;
	vector<int> correspondingNeighbourClassOne;
	vector<int> correspondingNeighbourClassZero;

	//message propagation
	//leaves to root
	vector<double> fi_ChildList;
	double fi[cNum];
	double fo[cNum];
	int foNode;
	int foNode_ischild;
};



struct Parameter {
	double Epsilon = 0.001;
	double Pi = 0.5;
	double rho = 0.999;
	double elnPi;
	double elnPiInv;
	double elnrho;
	double elnrhoInv;
	double elnHalf;

	vector<double> elnPzn_xn; // U-Net probability in log form
	double elnPz[cNum]; // prior class probability of a node without parents
	double elnPz_zpn[cNum][cNum]; // class transitional probability

	int allPixelSize = -1; //set a wrong default value, so it will report error if it's not initialized properly
	int ROW = -1;
	int COLUMN = -1;
	int regionId; // id of region being used
	int useHMT = 1; // whether or not to use HMT tree (SplitTree)
	float lambda; // regularization weight
	
	double Pi_orig = 0.5; // Pi is changed to log(Pi) but later we need Pi, so we have Pi_orig
	int split_threshold;
	int discard_NA_regions;
};

struct AdjacentNodePair{ 
	int currentNode;
	int adjacentNode;
	double currNodeCost;
	double adjNodeCost;

	AdjacentNodePair(int _currentNode, int _adjacentNode, double _currNodeCost, double _adjNodeCost)
		: currentNode(_currentNode), adjacentNode(_adjacentNode), currNodeCost(_currNodeCost), adjNodeCost(_adjNodeCost) {}
};

// data structure for adjacent regions
struct AdjacencyList{ 
	int regionId; // neighbor's region id
	vector<AdjacentNodePair> adjacentNodes;
	vector<int> to_sum; // store how much lambda to change for bottom (subtract) and top (add)
	vector<int> lambdas;

};

// data structure for each region
struct Regions{
	int regionId;
	int regionSize;

	vector<int> bfsOrder;
	vector<Node> allSortedNodes;
	int bfsRootNode = -1;

	double frontierCost;
	int frontierNodeIdx;

	double max_cost;

	// TODO: remove after test
	// vector<double> loglikelihood; 
	// vector<int> mainBranchNodeIds;

	vector<double> loglikelihoodUnique;
	vector<Node> sortedNodes;
	map<int, int> nodeId2Idx; // map between actual node_id and its index in a particular region

	double MIN_LOSS = INFINITY;
	
	vector<AdjacencyList> adjacencyList; // each element correspond to a neighboring region
	
	// for coloring
	int color;

	bool isAllFlood = false;
	bool isAllDry = false;
	bool isPartialFlood = false;
	bool tracesToRiver = false; // check if this region traces back to river through allFlood path

	// store original region id too
	int origRootId;

	// store parent and children regions
	int parentID;
	vector<int> childrenID;

	// for HMT Tree
	vector<pair<double, int>> costIndexPair;
	vector<int>sortedCostIndex;
	map<int, int> index2PixelId;

	// support for legacy code; remove later
	double inferredMaxCost;
	double inferredFrontierNodeIdx;

	Regions(int _regionId, int _bfsRootNode)
		: regionId(_regionId), bfsRootNode(_bfsRootNode) {}
};




struct Data {
	vector<Node>allNodes;
	vector<Regions> allRegions;
	int total_regions;

	vector<vector<int>> color_sets;
	vector<RegionVertex> regions; // for coloring

	vector<int> river_nodes; // pixels in river

	vector<int> regionIds; // find order of river nodes from upstream to downstream
	vector<int> largeRegionIds; // store large regions only
	map<int, int> regionId2Index; // get index of region based on region_id

	int** region_map; // store array for divided regions
	vector<int> index2NodeArray; // array to map index to new node id

	// for creating sub-trees
	vector<int> splitRegionIds;
	map<int, int> rootId2OrigRootId;

	// for convergence metrics
	atomic<int> curr_flipped_count;
	atomic<int> prev_flipped_count;
	atomic<int> curr_flood_count;
	atomic<int> prev_flood_count;

	//for log-likelihood; only process large regions in parallel
	vector<RegionVertex> large_regions;
	vector<RegionVertex> small_regions;
};


// used to create HMT tree structure
struct subset{
	int parent;
	int rank;
};




