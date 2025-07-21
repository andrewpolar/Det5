//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// he or she is under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://arxiv.org/abs/2305.08194

#include <iostream>
#include "KANAddendPL.h"
#include "Helper.h"

//determinants
std::unique_ptr<std::unique_ptr<double[]>[]> GenerateInput(int nRecords, int nFeatures, double min, double max) {
    auto x = std::make_unique<std::unique_ptr<double[]>[]>(nRecords);
    for (int i = 0; i < nRecords; ++i) {
        x[i] = std::make_unique<double[]>(nFeatures);
        for (int j = 0; j < nFeatures; ++j) {
            x[i][j] = static_cast<double>((rand() % 10000) / 10000.0);
            x[i][j] *= (max - min);
            x[i][j] += min;
        }
    }
    return x;
}

double determinant(const std::vector<std::vector<double>>& matrix) {
    int n = (int)matrix.size();
    if (n == 1) {
        return matrix[0][0];
    }
    if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    double det = 0.0;
    for (int col = 0; col < n; ++col) {
        std::vector<std::vector<double>> subMatrix(n - 1, std::vector<double>(n - 1));
        for (int i = 1; i < n; ++i) {
            int subCol = 0;
            for (int j = 0; j < n; ++j) {
                if (j == col) continue;
                subMatrix[i - 1][subCol++] = matrix[i][j];
            }
        }
        det += (col % 2 == 0 ? 1 : -1) * matrix[0][col] * determinant(subMatrix);
    }
    return det;
}

double ComputeDeterminant(std::unique_ptr<double[]>& input, int N) {
    std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = input[cnt++];
        }
    }
    return determinant(matrix);
}

std::unique_ptr<double[]> ComputeDeterminantTarget(const std::unique_ptr<std::unique_ptr<double[]>[]>& x, int nMatrixSize, int nRecords) {
    auto target = std::make_unique<double[]>(nRecords);
    int counter = 0;
    while (true) {
        target[counter] = ComputeDeterminant(x[counter], nMatrixSize);
        if (++counter >= nRecords) break;
    }
    return target;
}

void TestDeterminant_4_4() {
    //generate data
    int nTrainingRecords = 100000;
    int nValidationRecords = 20000;
    int nMatrixSize = 4;
    int nFeatures = nMatrixSize * nMatrixSize;
    double min = 0.0;
    double max = 1.0;
    auto inputs_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
    auto inputs_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
    auto target_training = ComputeDeterminantTarget(inputs_training, nMatrixSize, nTrainingRecords);
    auto target_validation = ComputeDeterminantTarget(inputs_validation, nMatrixSize, nValidationRecords);

    double reference = 0.0;
    for (int i = 0; i < nValidationRecords; ++i) {
        reference += target_validation[i] * target_validation[i];
    }

    //data is ready, we start training
    clock_t start_application = clock();
    clock_t current_time = clock();

    std::vector<double> argmin;
    std::vector<double> argmax;
    double targetMin;
    double targetMax;
    Helper::FindMinMax(argmin, argmax, targetMin, targetMax, inputs_training, target_training,
        nTrainingRecords, nFeatures);

    int nAddends = 66;
    int PWLEpochs = 50;
    double mu = 0.3;
    mu /= nAddends;

    auto xmin = std::make_unique<double[]>(argmin.size());
    auto xmax = std::make_unique<double[]>(argmax.size());
    for (int i = 0; i < argmin.size(); ++i) {
        xmin[i] = argmin[i];
        xmax[i] = argmax[i];
    }

    std::vector<std::unique_ptr<KANAddendPL>> addends;
    for (int i = 0; i < nAddends; ++i) {
        addends.push_back(std::make_unique<KANAddendPL>(xmin, xmax, targetMin / nAddends, targetMax / nAddends, 5, 22, nFeatures));
    }

    auto model_training = std::make_unique<double[]>(nTrainingRecords);
    auto model_validation = std::make_unique<double[]>(nValidationRecords);
    for (int epoch = 0; epoch < PWLEpochs; ++epoch) {
        double error = 0.0;
        for (int i = 0; i < nTrainingRecords; ++i) {
            double model = 0.0;
            for (int j = 0; j < nAddends; ++j) {
                model += addends[j]->ComputeUsingInput(inputs_training[i]);
            }
            model_training[i] = model;
            double residual = target_training[i] - model;
            error += residual * residual;
            residual *= mu;
            for (int j = 0; j < nAddends; ++j) {
                addends[j]->UpdateUsingMemory(residual);
            }
        }
        if (epoch >= 0) {
            double error = 0.0;
            for (int i = 0; i < nValidationRecords; ++i) {
                double vmodel = 0.0;
                for (int j = 0; j < nAddends; ++j) {
                    vmodel += addends[j]->ComputeUsingInput(inputs_validation[i], true);
                }
                model_validation[i] = vmodel;
                error += (target_validation[i] - vmodel) * (target_validation[i] - vmodel);
            }
            double L2 = sqrt(error / reference);
            error /= nValidationRecords;
            error = sqrt(error) / (targetMax - targetMin);
            double training_pearson = Helper::Pearson(model_training, target_training, nTrainingRecords);
            double validation_pearson =
                Helper::Pearson(model_validation, target_validation, nValidationRecords);
            current_time = clock();
            printf("E %d, training %6.3f, validation %6.3f, RRMSE %6.3f, L2 %6.3f, time %2.3f\n",
                epoch + 1, training_pearson, validation_pearson, error, L2,
                (double)(current_time - start_application) / CLOCKS_PER_SEC);
        }
        else {
            printf("Epoch %d of %d\r", epoch + 1, PWLEpochs);
        }
    }
}

void TestDeterminant_5_5() {
    //generate data
    int nTrainingRecords = 10000000;
    int nValidationRecords = 2000000;
    int nMatrixSize = 5;
    int nFeatures = nMatrixSize * nMatrixSize;
    double min = 0.0;
    double max = 10.0;
    auto inputs_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
    auto inputs_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
    auto target_training = ComputeDeterminantTarget(inputs_training, nMatrixSize, nTrainingRecords);
    auto target_validation = ComputeDeterminantTarget(inputs_validation, nMatrixSize, nValidationRecords);

    double reference = 0.0;
    for (int i = 0; i < nValidationRecords; ++i) {
        reference += target_validation[i] * target_validation[i];
    }

    //data is ready, we start training
    clock_t start_application = clock();
    clock_t current_time = clock();

    std::vector<double> argmin;
    std::vector<double> argmax;
    double targetMin;
    double targetMax;
    Helper::FindMinMax(argmin, argmax, targetMin, targetMax, inputs_training, target_training,
        nTrainingRecords, nFeatures);

    int nAddends = 200;
    int PWLEpochs = 10;
    double mu = 0.2;
    mu /= nAddends;

    auto xmin = std::make_unique<double[]>(argmin.size());
    auto xmax = std::make_unique<double[]>(argmax.size());
    for (int i = 0; i < argmin.size(); ++i) {
        xmin[i] = argmin[i];
        xmax[i] = argmax[i];
    }

    std::vector<std::unique_ptr<KANAddendPL>> addends;
    for (int i = 0; i < nAddends; ++i) {
        addends.push_back(std::make_unique<KANAddendPL>(xmin, xmax, targetMin / nAddends, targetMax / nAddends, 5, 22, nFeatures));
    }

    auto model_training = std::make_unique<double[]>(nTrainingRecords);
    auto model_validation = std::make_unique<double[]>(nValidationRecords);
    for (int epoch = 0; epoch < PWLEpochs; ++epoch) {
        double error = 0.0;
        for (int i = 0; i < nTrainingRecords; ++i) {
            double model = 0.0;
            for (int j = 0; j < nAddends; ++j) {
                model += addends[j]->ComputeUsingInput(inputs_training[i]);
            }
            model_training[i] = model;
            double residual = target_training[i] - model;
            error += residual * residual;
            residual *= mu;
            for (int j = 0; j < nAddends; ++j) {
                addends[j]->UpdateUsingMemory(residual);
            }
        }
        if (epoch >= 0) {
            double error = 0.0;
            for (int i = 0; i < nValidationRecords; ++i) {
                double vmodel = 0.0;
                for (int j = 0; j < nAddends; ++j) {
                    vmodel += addends[j]->ComputeUsingInput(inputs_validation[i], true);
                }
                model_validation[i] = vmodel;
                error += (target_validation[i] - vmodel) * (target_validation[i] - vmodel);
            }
            double L2 = sqrt(error / reference);
            error /= nValidationRecords;
            error = sqrt(error) / (targetMax - targetMin);
            double training_pearson = Helper::Pearson(model_training, target_training, nTrainingRecords);
            double validation_pearson =
                Helper::Pearson(model_validation, target_validation, nValidationRecords);
            current_time = clock();
            printf("E %d, training %6.3f, validation %6.3f, RRMSE %6.3f, L2 %6.3f, time %2.3f\n",
                epoch + 1, training_pearson, validation_pearson, error, L2,
                (double)(current_time - start_application) / CLOCKS_PER_SEC);
        }
        else {
            printf("Epoch %d of %d\r", epoch + 1, PWLEpochs);
        }
    }
}

int main()
{
    srand((unsigned int)time(NULL));
    TestDeterminant_4_4();
    //Next function needs at least 30 minutes
    //TestDeterminant_5_5();
}

