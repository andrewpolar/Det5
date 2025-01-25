#include "KANAddendPL.h"
#include <memory>

KANAddendPL::KANAddendPL(std::unique_ptr<double[]>& xmin, std::unique_ptr<double[]>& xmax,
    double targetMin, double targetMax,
    int inner, int outer, int number_of_inputs) {

    _targetMin = targetMin;
    _targetMax = targetMax;
    _lastInnerValue = 0.0;

    std::unique_ptr<int[]> interior_structure = std::make_unique<int[]>(number_of_inputs);
    for (int i = 0; i < number_of_inputs; i++)
    {
        interior_structure[i] = static_cast<int>(inner);
    }
    _u = std::make_unique<UrysohnPL>(xmin, xmax, _targetMin, _targetMax, interior_structure, number_of_inputs);
    _univariate = std::make_unique<UnivariatePL>(_targetMin, _targetMax, _targetMin, _targetMax, outer);
}

KANAddendPL::~KANAddendPL() {
}

KANAddendPL::KANAddendPL(const KANAddendPL& addend) {
    _targetMin = addend._targetMin;
    _targetMax = addend._targetMax;
    _lastInnerValue = addend._lastInnerValue;
    _univariate = std::make_unique<UnivariatePL>(*addend._univariate);
    _u = std::make_unique<UrysohnPL>(*addend._u);
}

void KANAddendPL::IncrementInner() {
    _u->IncrementInner();
}

int KANAddendPL::HowManyOuter() {
    return _univariate->HowManyPoints();
}

int KANAddendPL::HowManyInner() {
    return _u->_univariateList[0]->HowManyPoints();
}

std::unique_ptr<double[]> KANAddendPL::GetAllOuterPoints() {
    return _univariate->GetAllPoints();
}

void KANAddendPL::IncrementOuter() {
    _univariate->IncrementPoints();
}

void KANAddendPL::UpdateUsingMemory(double diff) {
    double derrivative = _univariate->GetDerivative(_lastInnerValue);
    _u->UpdateUsingMemory(diff * derrivative);
    _univariate->UpdateUsingMemory(diff);
}

void KANAddendPL::UpdateUsingInput(std::unique_ptr<double[]>& input, double diff) {
    double value = _u->GetValueUsingInput(input);
    double derrivative = _univariate->GetDerivative(value);
    _u->UpdateUsingInput(diff * derrivative, input);
    _univariate->UpdateUsingInput(value, diff);
}

double KANAddendPL::ComputeUsingInput(std::unique_ptr<double[]>& input, bool noUpdate) {
    _lastInnerValue = _u->GetValueUsingInput(input, noUpdate);
    return _univariate->GetFunctionUsingInput(_lastInnerValue, noUpdate);
}
