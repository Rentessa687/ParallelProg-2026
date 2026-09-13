// jacobi.cpp
// Последовательная реализация метода Якоби для решения СЛАУ Ax = b.
// Это baseline-версия (без распараллеливания) для лабораторной работы №1.
// В последующих лабораторных работах данный алгоритм будет распараллелен
// с помощью OpenMP / MPI и т.д.
//
// Формат входного файла:
//   n
//   a11 a12 ... a1n
//   a21 a22 ... a2n
//   ...
//   an1 an2 ... ann
//   b1 b2 ... bn
//
// Формат выходного файла:
//   n
//   x1 x2 ... xn
//   iterations <k>
//   time_ms <t>
//   residual <r>
//   eps <e>
//   problem_size <n>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

static bool readInput(const std::string &filename, int &n,
                       std::vector<double> &A, std::vector<double> &b) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Ошибка: не удалось открыть входной файл " << filename << std::endl;
        return false;
    }
    fin >> n;
    if (n <= 0) {
        std::cerr << "Ошибка: некорректный размер системы n = " << n << std::endl;
        return false;
    }
    A.resize((size_t)n * (size_t)n);
    b.resize(n);
    for (int i = 0; i < n * n; ++i) {
        if (!(fin >> A[i])) {
            std::cerr << "Ошибка: не удалось прочитать элемент матрицы A[" << i << "]" << std::endl;
            return false;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (!(fin >> b[i])) {
            std::cerr << "Ошибка: не удалось прочитать элемент вектора b[" << i << "]" << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0]
                   << " <input_file> [output_file] [eps] [max_iter]" << std::endl;
        return 1;
    }
    const std::string input_file = argv[1];
    const std::string output_file = (argc >= 3) ? argv[2] : "output.txt";
    const double eps = (argc >= 4) ? std::atof(argv[3]) : 1e-6;
    const int max_iter = (argc >= 5) ? std::atoi(argv[4]) : 100000;

    int n;
    std::vector<double> A, b;
    if (!readInput(input_file, n, A, b)) return 1;

    // Проверка диагональных элементов (метод Якоби требует a_ii != 0)
    for (int i = 0; i < n; ++i) {
        if (std::fabs(A[(size_t)i * n + i]) < 1e-14) {
            std::cerr << "Ошибка: нулевой диагональный элемент A[" << i << "][" << i << "]" << std::endl;
            return 1;
        }
    }

    std::vector<double> x(n, 0.0), x_new(n, 0.0);

    auto start = std::chrono::high_resolution_clock::now();

    int iter = 0;
    double diff = 0.0;
    for (iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            const double* Ai = &A[(size_t)i * n];
            for (int j = 0; j < n; ++j) {
                if (j != i) sigma += Ai[j] * x[j];
            }
            x_new[i] = (b[i] - sigma) / Ai[i];
        }

        diff = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = x_new[i] - x[i];
            diff += d * d;
        }
        diff = std::sqrt(diff);

        x.swap(x_new);

        if (diff < eps) {
            ++iter;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Невязка ||Ax - b||_2 — показатель качества решения
    double residual = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double* Ai = &A[(size_t)i * n];
        for (int j = 0; j < n; ++j) s += Ai[j] * x[j];
        double r = s - b[i];
        residual += r * r;
    }
    residual = std::sqrt(residual);

    std::ofstream fout(output_file);
    if (!fout) {
        std::cerr << "Ошибка: не удалось открыть выходной файл " << output_file << std::endl;
        return 1;
    }
    fout.precision(12);
    fout << n << "\n";
    for (int i = 0; i < n; ++i) {
        fout << x[i] << (i + 1 < n ? ' ' : '\n');
    }
    fout << "iterations " << iter << "\n";
    fout << "time_ms " << elapsed_ms << "\n";
    fout << "residual " << residual << "\n";
    fout << "eps " << eps << "\n";
    fout << "problem_size " << n << "\n";

    std::cout << "Метод Якоби завершён.\n";
    std::cout << "n = " << n << ", итераций = " << iter
              << ", время = " << elapsed_ms << " мс\n";
    std::cout << "невязка ||Ax-b|| = " << residual << "\n";
    if (diff >= eps && iter >= max_iter) {
        std::cout << "ВНИМАНИЕ: достигнут лимит итераций (max_iter), сходимость не гарантирована.\n";
    }
    std::cout << "Результат записан в " << output_file << std::endl;

    return 0;
}
