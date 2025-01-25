#pragma once
#include <memory>
#include <vector>

class Helper
{
public:
    static double Pearson(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, int len) {
        double xy = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;
        for (int i = 0; i < len; ++i)
        {
            xy += x[i] * y[i];
            x2 += x[i] * x[i];
            y2 += y[i] * y[i];
        }
        xy /= (double)(len);
        x2 /= (double)(len);
        y2 /= (double)(len);
        double xav = 0.0;
        for (int i = 0; i < len; ++i)
        {
            xav += x[i];
        }
        xav /= len;
        double yav = 0.0;
        for (int i = 0; i < len; ++i)
        {
            yav += y[i];
        }
        yav /= len;
        double ro = xy - xav * yav;
        ro /= sqrt(x2 - xav * xav);
        ro /= sqrt(y2 - yav * yav);
        return ro;
    }
    //this gives identical result
    static double Pearson2(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, int len) {
        double xmean = 0.0;
        double ymean = 0.0;
        for (int i = 0; i < len; ++i) {
            xmean += x[i];
            ymean += y[i];
        }
        xmean /= len;
        ymean /= len;

        double covariance = 0.0;
        for (int i = 0; i < len; ++i) {
            covariance += (x[i] - xmean) * (y[i] - ymean);
        }

        double stdX = 0.0;
        double stdY = 0.0;
        for (int i = 0; i < len; ++i) {
            stdX += (x[i] - xmean) * (x[i] - xmean);
            stdY += (y[i] - ymean) * (y[i] - ymean);
        }
        stdX = sqrt(stdX);
        stdY = sqrt(stdY);
        return covariance / stdX / stdY;
    }
    static void ShowMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%5.3f ", matrix[i][j]);
            }
            printf("\n");
        }
    }
    static void ShowVector(std::unique_ptr<double[]>& ptr, int N) {
        int cnt = 0;
        for (int i = 0; i < N; ++i) {
            printf("%5.2f ", ptr[i]);
            if (++cnt >= 10) {
                printf("\n");
                cnt = 0;
            }
        }
    }
    static void SwapRows(std::unique_ptr<double[]>& row1, std::unique_ptr<double[]>& row2, int cols) {
        auto ptr = std::make_unique<double[]>(cols);
        for (int i = 0; i < cols; ++i) {
            ptr[i] = row1[i];
        }
        for (int i = 0; i < cols; ++i) {
            row1[i] = row2[i];
        }
        for (int i = 0; i < cols; ++i) {
            row2[i] = ptr[i];
        }
    }
    static void SwapScalars(double& x1, double& x2) {
        double buff = x1;
        x1 = x2;
        x2 = buff;
    }
    static void Shuffle(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, std::unique_ptr<double[]>& vector, int rows, int cols) {
        for (int i = 0; i < 2 * rows; ++i) {
            int n1 = rand() % rows;
            int n2 = rand() % rows;
            SwapRows(matrix[n1], matrix[n2], cols);
            SwapScalars(vector[n1], vector[n2]);
        }
    }
    static void FindMinMax(std::vector<double>& xmin, std::vector<double>& xmax,
        double& targetMin, double& targetMax,
        std::unique_ptr<std::unique_ptr<double[]>[]>& matrix,
        std::unique_ptr<double[]>& target, int nRows, int nCols) {

        for (int i = 0; i < nCols; ++i) {
            xmin.push_back(DBL_MAX);
            xmax.push_back(-DBL_MAX);
        }
        for (int i = 0; i < nRows; ++i) {
            for (int j = 0; j < nCols; ++j) {
                if (matrix[i][j] < xmin[j]) xmin[j] = static_cast<double>(matrix[i][j]);
                if (matrix[i][j] > xmax[j]) xmax[j] = static_cast<double>(matrix[i][j]);
            }
        }
        targetMin = DBL_MAX;
        targetMax = -DBL_MAX;
        for (int j = 0; j < nRows; ++j) {
            if (target[j] < targetMin) targetMin = target[j];
            if (target[j] > targetMax) targetMax = target[j];
        }
    }
    static void IndividualLimits2Sum(double xmin, double xmax, int N, double& sumMin, double& sumMax) {
        double sumMean = N * (xmin + xmax) / 2.0;
        double sumVariance = N * (xmin - xmax) * (xmin - xmax) / 12.0;
        double STD = sqrt(sumVariance);
        sumMin = sumMean - 1.96 * STD;
        sumMax = sumMean + 1.96 * STD;
    }
    static void Sum2IndividualLimits(double sumMin, double sumMax, int N, double& xmin, double& xmax) {
        double varCoeff = N / 12.0;
        double STDCoeff = sqrt(varCoeff);
        double confCoeff = STDCoeff * 1.96;
        double xmin_plus_xmax = (sumMin + sumMax) / N;
        double xmax_minus_xmin = (sumMax - sumMin) / 2 / confCoeff;
        xmax = (xmin_plus_xmax + xmax_minus_xmin) / 2.0;
        xmin = xmin_plus_xmax - xmax;
    }
};

