#pragma unmanaged

#include "GrowCut.h"

#include <iostream>


GrowCut::GrowCut(const std::vector<int>& sourceReso, int labelTypeNum)
{
	if (labelTypeNum < 1) throw std::exception();

	prohibit_attack_threshold_ = 2 * (int)sourceReso.size(); 
	force_occupied_threshold_  = 2 * (int)sourceReso.size();

	label_type_num_ = labelTypeNum;
	for (size_t i = 0; i < sourceReso.size(); ++i)
		src_resolution_.push_back(sourceReso[i]);
}


GrowCut::GrowCut
( const std::vector<int>& sourceReso
, int labelTypeNum
, int prohibitAttackThres
, int forceOccupiedThres
)
{
	if (labelTypeNum < 1) throw std::exception();

	prohibit_attack_threshold_ = prohibitAttackThres;
	force_occupied_threshold_  = forceOccupiedThres;

	label_type_num_ = labelTypeNum;
	for (size_t i = 0; i < sourceReso.size(); ++i)
		src_resolution_.push_back(sourceReso[i]);
}



GrowCut::~GrowCut()
{
}


void GrowCut::initState(const short* src, const int* initLabel)
{
	src_size_ = 1;
	for (const auto& reso : src_resolution_)
		src_size_ *= reso;
	src_state_.resize(src_size_);
#pragma omp parallel for
	for (int i = 0; i < src_size_; ++i)
	{
    EVecXi v(1);
    v << (int)src[i];
		src_state_[i] = CellState(initLabel[i], 1.0, v);
		if (initLabel[i] < 0) src_state_[i].strength_ = 0.0;
	}
}


void GrowCut::Segmentation(const short* src, const int* initLabel, std::vector<int>& result)
{
	initState(src, initLabel);
	int stepI = 0;
	while( GrowCell() ) std::cout << "Grow Cell : "<< stepI++ << std::endl;

	result.clear();
	result.resize(src_size_); 
#pragma omp parallel for
	for (int i = 0; i < (int)src_state_.size(); ++i)
		result[i] = src_state_[i].label_;
}


bool GrowCut::SegmentationOneStep(const short* src, const int* initLabel, std::vector<int>& result, bool isInit)
{
	if (isInit) initState(src, initLabel);

	bool isChangeState = GrowCell();

	result.clear();
	result.resize(src_size_);
#pragma omp parallel for
	for (int i = 0; i < (int)src_state_.size(); ++i)
		result[i] = src_state_[i].label_;

	return isChangeState;
}



bool GrowCut::GrowCell()
{
	bool isChangeState = false;

#pragma omp parallel for
	for (int i = 0; i < src_size_; ++i) if (src_state_[i].label_ > -2)
	{
		std::vector<int> neighborIndices = searchNeighborIdx(i);
		if ( IsForceOccupied(i, neighborIndices) )
		{	
			isChangeState = true;
			continue;
		}

		//近傍セルが注目セルに攻撃を仕掛ける
		for ( const auto& neighbor : neighborIndices )
		{
			//近傍セルの近傍に敵がたくさんいる場合は攻撃を禁止
			int weakestEnemyIdx;
			int maxEnemyNum = calcMaxEnemyNum(neighbor, searchNeighborIdx(neighbor), weakestEnemyIdx);
			if (maxEnemyNum >= prohibit_attack_threshold_) continue;

			double edge = EvaluationFunc(i, neighbor) * src_state_[neighbor].strength_;
			if (edge > src_state_[i].strength_)
			{
				src_state_[i].label_ = src_state_[neighbor].label_;
				src_state_[i].strength_ = edge;
				isChangeState = true;
			}
		}
	}

	return isChangeState;
}

// ノイマン近傍
std::vector<int> GrowCut::searchNeighborIdx(int idx)
{
	std::vector<int> neighborIndices;
	for (int i = 0; i < (int)src_resolution_.size(); ++i)
	{
		int neighbor  = 1;
		for (int j = 0; j < i; ++j)
			neighbor *= src_resolution_[j];

		//変更不可ラベルが定義されているセルは近傍とみなさない
		if (idx + neighbor < src_size_ && src_state_[idx + neighbor].label_ > -2) 
			neighborIndices.push_back(idx + neighbor);

		if (idx - neighbor >= 0 && src_state_[idx - neighbor].label_ > -2) 
			neighborIndices.push_back(idx - neighbor);
	}

	return neighborIndices;
}


//[0,1]に正規化されており、かつ、色差が大きいほど評価値が小さくなるように関数
double GrowCut::EvaluationFunc(int idx1, int idx2)
{
	return exp( -(src_state_[idx1].feature_vec_ - src_state_[idx2].feature_vec_).norm() );
}


int GrowCut::calcMaxEnemyNum(int idx, const std::vector<int>& neighborIndices, int& weakestEnemyIdx)
{
	std::vector<int> enemyNum(label_type_num_, 0);
	for (const auto& neighbor : neighborIndices)
	{
		int neighborLabel = src_state_[neighbor].label_;
		//近傍セルのラベルが定義済み　かつ　注目セルのラベルと違う場合
		if (neighborLabel >= 0 && src_state_[idx].label_ != neighborLabel)
			enemyNum[neighborLabel]++;
	}

	int maxEnemyNum = 0;
	int maxEnemyLabel = 0;
	for (int label = 0; label < (int)enemyNum.size(); ++label)
	if (enemyNum[label] > maxEnemyNum)
	{
		maxEnemyNum = enemyNum[label];
		maxEnemyLabel = label;
	}

	double minStrength = DBL_MAX;
	for (const int& neighbor : neighborIndices)
	if (src_state_[neighbor].label_ == maxEnemyLabel && src_state_[neighbor].strength_ < minStrength)
	{
		minStrength = src_state_[neighbor].strength_;
		weakestEnemyIdx = neighbor;
	}
	
	return maxEnemyNum;
}


bool GrowCut::IsForceOccupied(int interestIdx, const std::vector<int>& neighborIndices)
{
	int weakestEnemyIdx;
	int maxEnemyNum = calcMaxEnemyNum(interestIdx, neighborIndices, weakestEnemyIdx);

	//注目セルの近傍に多くの敵がいる場合は、最も弱い近傍セルに侵略される
	if (maxEnemyNum >= force_occupied_threshold_)
	{
		double newStrength = EvaluationFunc(interestIdx, weakestEnemyIdx) 
											 * src_state_[weakestEnemyIdx].strength_;

		src_state_[interestIdx].label_ = src_state_[weakestEnemyIdx].label_;
		src_state_[interestIdx].strength_ = newStrength;
		
		return true;
	}
	else
		return false;
}

#pragma managed