#include "UrysohnPL.h"
#include "Helper.h"
#include <cstdlib> 
#include <iostream>

UrysohnPL::UrysohnPL(std::unique_ptr<double[]>& xmin, std::unique_ptr<double[]>& xmax,
	double targetMin, double targetMax,
	std::unique_ptr<int[]>& layers, int len) {
	_length = len;
	double ymin = targetMin / _length;
	double ymax = targetMax / _length;
	Helper::Sum2IndividualLimits(targetMin, targetMax, len, ymin, ymax);
	_univariateList = std::make_unique<std::unique_ptr<UnivariatePL>[]>(_length);
	for (int i = 0; i < _length; ++i) {
		_univariateList[i] = std::make_unique<UnivariatePL>(xmin[i], xmax[i], ymin, ymax, layers[i]);
	}
}

UrysohnPL::~UrysohnPL() {
}

UrysohnPL::UrysohnPL(const UrysohnPL& uri) {
	_length = uri._length;
	_univariateList = std::make_unique<std::unique_ptr<UnivariatePL>[]>(_length);
	for (int i = 0; i < _length; ++i) {
		_univariateList[i] = std::make_unique<UnivariatePL>(*uri._univariateList[i]);
	}
}

void UrysohnPL::IncrementInner() {
	for (int i = 0; i < _length; ++i) {
		_univariateList[i]->IncrementPoints();
	}
}

void UrysohnPL::UpdateUsingInput(double delta, std::unique_ptr<double[]>& inputs) {
	for (int i = 0; i < _length; ++i) {
		_univariateList[i]->UpdateUsingInput(inputs[i], delta);
	}
}

void UrysohnPL::UpdateUsingMemory(double delta) {
	for (int i = 0; i < _length; ++i) {
		_univariateList[i]->UpdateUsingMemory(delta);
	}
}

double UrysohnPL::GetValueUsingInput(std::unique_ptr<double[]>& inputs, bool noUpdate) {
	double f = 0.0;
	for (int i = 0; i < _length; ++i) {
		f += _univariateList[i]->GetFunctionUsingInput(inputs[i], noUpdate);
	}
	return f;
}

std::unique_ptr<double[]> UrysohnPL::GetUPoints(int n) {
	return _univariateList[n]->GetAllPoints();
}

