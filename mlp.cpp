#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
using namespace std;

double dtanh(double x) {
    double tanh_x = tanh(x); // use tanh function, not sigmoid function !
    return 1 - tanh_x * tanh_x;   // tanh derivative
}

double init_weight() {
    return ((double)rand() / RAND_MAX) * 0.1;  // initial -0.5 ~ 0.5
}

void shuffle(int *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main() {
    static const int numInputs = 1;
    static const int numHiddenNodes = 21;
    static const int numOutputs = 1;

    const double lr = 0.2;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    static const int numTrainingSets = 21;
    double training_inputs[numTrainingSets][numInputs] = { {0}, {0.2}, {0.4}, {0.6}, {0.8}, {1}, {1.2}, {1.4}, {1.6}, {1.8}, {2}, {2.2}, {2.4}, {2.6}, {2.8}, {3}, {3.2}, {3.4}, {3.6}, {3.8}, {4} };
    double training_outputs[numTrainingSets][numOutputs] = { {1}, {0.675728}, {0.242896}, {-0.12469}, {-0.33133}, {-0.3642}, {-0.2701}, {-0.1209}, {0.017666}, {0.104914}, {0.129945}, {0.105289}, {0.055188}, {0.004007}, {-0.03158}, {-0.04536}, {-0.04014}, {-0.02384}, {-0.00531}, {0.008803}, {0.015456} };

    // initial weight and bias
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            hiddenWeights[i][j] = init_weight();
        }
    }
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight();
        for (int j = 0; j < numOutputs; j++) {
            outputWeights[i][j] = init_weight();
        }
    }
    for (int i = 0; i < numOutputs; i++) {
        outputLayerBias[i] = init_weight();
    }

    int trainingSetOrder[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

    int outputCount = 0;

    for (int n = 0; n < 30000; n++) {
        shuffle(trainingSetOrder, numTrainingSets);
        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];

            // Forward propagation
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = tanh(activation);
            }

            for (int j = 0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = tanh(activation);
            }

            // run 100 time
            if (outputCount < 100) {
                cout << fixed << setprecision(6);
                cout <<"Iter: "<<"["<<outputCount+1<<"]"<< " input: " << training_inputs[i][0]<< "   Predicted Value: " << outputLayer[0]<< "   Actual Value: " << training_outputs[i][0] << endl;
                outputCount++;
            }

            // Backward propagation
            double deltaOutput[numOutputs];
            for (int j = 0; j < numOutputs; j++) {
                double errorOutput = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = errorOutput * dtanh(outputLayer[j]);
            }

            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for (int k = 0; k < numOutputs; k++) {
                    errorHidden += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = errorHidden * dtanh(hiddenLayer[j]);
            }

            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }
    system("pause");
    return 0;
}
