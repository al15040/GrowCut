#pragma once

#include<vector>
#include"Core"

typedef Eigen::VectorXi EVecXi;

struct CellState
{
	int label_; // -1 : undefined, -2 // �ύX�s��
	double strength_;
	EVecXi feature_vec_;

	CellState(int label, double strength, const EVecXi& featureVec)
	{
		label_ = label;
		strength_ = strength;
		feature_vec_ = featureVec;
	}
};


class GrowCut
{
private:
	std::vector<int> src_resolution_;
	std::vector<CellState> src_state_;
	int src_size_;
	int label_type_num_;

	int prohibit_attack_threshold_; //�U�����֎~����臒l
	int force_occupied_threshold_; //�����I�ɐN�H�����臒l

public:
	GrowCut(const std::vector<int>& sourceReso, int labelTypeNum);
	GrowCut(const std::vector<int>& sourceReso, int labelTypeNum, int prohibitAttackThres, int forceOccupiedThres);
	~GrowCut();

	void Segmentation(const short* src, const int* initLabel, std::vector<int>& result);
private:
	void initState(const short* src, const int* initLabel);
	bool GrowCell();
	std::vector<int> searchNeighborIdx(int idx);
	double EvaluationFunc(int idx1, int idx2);
	int calcMaxEnemyNum(int idx, const std::vector<int>& neighborIdx, int& weakestEnemyIdx);

	bool IsForceOccupied(int interestIdx);
};