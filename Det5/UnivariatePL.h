#pragma once
#include <cstddef>
#include <memory>

class UnivariatePL
{
public:
    UnivariatePL(double xmin, double xmax, double ymin, double ymax, int points);
    ~UnivariatePL();
    UnivariatePL(const UnivariatePL& uni);
    int _points;
    double _xmin;
    double _xmax;
    double GetDerivative(double x);
    void UpdateUsingInput(double x, double delta);
    void UpdateUsingMemory(double delta);
    double GetFunctionUsingInput(double x, bool noUpdates = false);
    void IncrementPoints();
    int HowManyPoints();
    std::unique_ptr<double[]> GetAllPoints();
private:
    std::unique_ptr<double[]> _y;
    double _deltax;
    int _lastLeftIndex;
    double _lastLeftOffset;
    void SetLimits();
    void SetRandomFunction(double ymin, double ymax);
    void FitDefinition(double x);
};
