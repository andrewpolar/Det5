#pragma once
#include "UnivariatePL.h"
#include <memory>

class UrysohnPL
{
public:
	int _length;
	std::unique_ptr<std::unique_ptr<UnivariatePL>[]> _univariateList;
	UrysohnPL(std::unique_ptr<double[]>& xmin, std::unique_ptr<double[]>& xmax,
		double targetMin, double targetMax,
		std::unique_ptr<int[]>& layers, int len);
	~UrysohnPL();
	UrysohnPL(const UrysohnPL& uri);
	void UpdateUsingInput(double delta, std::unique_ptr<double[]>& inputs);
	void UpdateUsingMemory(double delta);
	double GetValueUsingInput(std::unique_ptr<double[]>& inputs, bool noUpdate = false);
	void IncrementInner();
	std::unique_ptr<double[]> GetUPoints(int n);
};

