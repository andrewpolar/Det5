#pragma once
#include "UrysohnPL.h"
#include "UnivariatePL.h"
#include <memory>

class KANAddendPL
{
public:
    double _targetMin;
    double _targetMax;
    KANAddendPL(std::unique_ptr<double[]>& xmin, std::unique_ptr<double[]>& xmax,
        double targetMin, double targetMax,
        int inner, int outer, int number_of_inputs);
    KANAddendPL(const KANAddendPL& addend);
    ~KANAddendPL();
    void UpdateUsingMemory(double diff);
    void UpdateUsingInput(std::unique_ptr<double[]>& input, double diff);
    double ComputeUsingInput(std::unique_ptr<double[]>& input, bool noUpdate = false);
    void IncrementInner();
    void IncrementOuter();
    int HowManyOuter();
    int HowManyInner();
    std::unique_ptr<double[]> GetAllOuterPoints();
    std::unique_ptr<UrysohnPL> _u;
private:
    double _lastInnerValue;
    std::unique_ptr<UnivariatePL> _univariate;
};

