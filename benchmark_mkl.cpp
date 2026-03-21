#include <Eigen/Dense>
#include <iostream>
#include <chrono>

using namespace std;
using namespace Eigen;

int main() {

    int N = 1000;   // same size as your Hamiltonian
    MatrixXd A = MatrixXd::Random(N,N);
    MatrixXd B = MatrixXd::Random(N,N);

    auto start = chrono::high_resolution_clock::now();

    for(int i=0;i<200;i++) {
        MatrixXd C = A * B;
    }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end-start;

    cout << "Runtime: " << elapsed.count() << " seconds" << endl;

}
