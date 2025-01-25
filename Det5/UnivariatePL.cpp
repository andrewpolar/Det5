#include "UnivariatePL.h"
#include <cstdlib> 

UnivariatePL::UnivariatePL(double xmin, double xmax, double ymin, double ymax, int points) {
    _points = points;
    _xmin = xmin;
    _xmax = xmax;
    SetLimits();
    SetRandomFunction(ymin, ymax);
}

UnivariatePL::~UnivariatePL() {
}

UnivariatePL::UnivariatePL(const UnivariatePL& uni) {
    _points = uni._points;
    _xmin = uni._xmin;
    _xmax = uni._xmax;
    _deltax = (_xmax - _xmin) / (_points - 1);
    _y.reset(nullptr);
    _y = std::make_unique<double[]>(_points);
    for (int i = 0; i < _points; i++)
    {
        _y[i] = uni._y[i];
    }
    _lastLeftIndex = uni._lastLeftIndex;
    _lastLeftOffset = uni._lastLeftOffset;
}

int UnivariatePL::HowManyPoints() {
    return _points;
}

void UnivariatePL::IncrementPoints() {
    int points = _points + 1;
    double deltax = (_xmax - _xmin) / (points - 1);
    auto y = std::make_unique<double[]>(points);
    y[0] = _y[0];
    y[points - 1] = _y[_points - 1];
    for (int i = 1; i < points - 1; ++i) {
        y[i] = GetFunctionUsingInput(_xmin + i * deltax);
    }
    _points = points;
    _deltax = deltax;
    _y.reset(nullptr);
    _y = std::make_unique<double[]>(_points);
    for (int i = 0; i < _points; i++)
    {
        _y[i] = y[i];
    }
}

void UnivariatePL::SetLimits() {
    double range = _xmax - _xmin;
    _xmin -= 0.01 * range;
    _xmax += 0.01 * range;
    _deltax = (_xmax - _xmin) / (_points - 1);
}

void UnivariatePL::SetRandomFunction(double ymin, double ymax) {
    _y = std::make_unique<double[]>(_points);
    for (int i = 0; i < _points; ++i)
    {
        _y[i] = (rand() % 100 / 100.0 * (ymax - ymin) + ymin);
    }
}

void UnivariatePL::FitDefinition(double x) {
    if (x < _xmin) {
        _xmin = x;
        SetLimits();
    }
    if (x > _xmax) {
        _xmax = x;
        SetLimits();
    }
}

double UnivariatePL::GetDerivative(double x) {
    int low = (int)((x - _xmin) / _deltax);
    return (_y[low + 1] - _y[low]) / _deltax;
}

void UnivariatePL::UpdateUsingInput(double x, double delta) {
    FitDefinition(x);
    double offset = (x - _xmin) / _deltax;
    int left = (int)(offset);
    double leftx = offset - left;
    double tmp = delta * leftx;
    _y[left + 1] += tmp;
    _y[left] += delta - tmp;
}

void UnivariatePL::UpdateUsingMemory(double delta) {
    double tmp = delta * _lastLeftOffset;
    _y[_lastLeftIndex + 1] += tmp;
    _y[_lastLeftIndex] += delta - tmp;
}

double UnivariatePL::GetFunctionUsingInput(double x, bool noUpdates) {
    if (noUpdates) {
        if (x < _xmin) x = _xmin;
        if (x > _xmax) x = _xmax;
    }
    else {
        FitDefinition(x);
    }
    double offset = (x - _xmin) / _deltax;
    int leftIndex = (int)(offset);
    double leftOffset = offset - leftIndex;
    _lastLeftIndex = leftIndex;
    _lastLeftOffset = leftOffset;
    return _y[leftIndex] + (_y[leftIndex + 1] - _y[leftIndex]) * leftOffset;
}

std::unique_ptr<double[]> UnivariatePL::GetAllPoints() {
    auto x = std::make_unique<double[]>(_points);
    for (int i = 0; i < _points; ++i) {
        x[i] = _y[i];
    }
    return x;
}
