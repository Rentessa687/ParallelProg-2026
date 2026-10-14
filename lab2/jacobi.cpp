#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    for (size_t i = 0; i < (size_t)n * (size_t)n; ++i) {
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

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0]
                  << " <input_file> [output_file] [eps] [max_iter] [threads]" << std::endl;
        return 1;
    }
    const std::string input_file = argv[1];
    const std::string output_file = (argc >= 3) ? argv[2] : "output.txt";
    const double eps = (argc >= 4) ? std::atof(argv[3]) : 1e-6;
    const int max_iter = (argc >= 5) ? std::atoi(argv[4]) : 100000;
    const int requested_threads = (argc >= 6) ? std::atoi(argv[5]) : 0;

#ifdef _OPENMP
    if (requested_threads > 0) omp_set_num_threads(requested_threads);
    const int used_threads = omp_get_max_threads();
#else
    (void)requested_threads;
    const int used_threads = 1;
#endif

    int n;
    std::vector<double> A, b;
    if (!readInput(input_file, n, A, b)) return 1;

    for (int i = 0; i < n; ++i) {
        if (std::fabs(A[(size_t)i * n + i]) < 1e-14) {
            std::cerr << "Ошибка: нулевой диагональный элемент A[" << i << "][" << i << "]" << std::endl;
            return 1;
        }
    }

    std::vector<double> x_buf(n, 0.0), x_new_buf(n, 0.0);
    const double *Ap = A.data();
    const double *bp = b.data();
    double *x = x_buf.data();
    double *x_new = x_new_buf.data();

    auto start = std::chrono::high_resolution_clock::now();

    int iter = 0;
    double diff = 0.0;
    for (iter = 0; iter < max_iter; ++iter) {
        double diff_sq = 0.0;

#pragma omp parallel for reduction(+ : diff_sq) schedule(static)
        for (int i = 0; i < n; ++i) {
            const double *Ai = Ap + (size_t)i * n;
            double sigma = 0.0;
            for (int j = 0; j < i; ++j) sigma += Ai[j] * x[j];
            for (int j = i + 1; j < n; ++j) sigma += Ai[j] * x[j];
            const double xi = (bp[i] - sigma) / Ai[i];
            x_new[i] = xi;
            const double d = xi - x[i];
            diff_sq += d * d;
        }

        diff = std::sqrt(diff_sq);
        std::swap(x, x_new);

        if (diff < eps) {
            ++iter;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    double residual = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        const double *Ai = Ap + (size_t)i * n;
        for (int j = 0; j < n; ++j) s += Ai[j] * x[j];
        const double r = s - bp[i];
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
    fout << "threads " << used_threads << "\n";

    std::cout << "Метод Якоби завершён.\n";
    std::cout << "n = " << n << ", потоков = " << used_threads
              << ", итераций = " << iter << ", время = " << elapsed_ms << " мс\n";
    std::cout << "невязка ||Ax-b|| = " << residual << "\n";
    if (diff >= eps && iter >= max_iter) {
        std::cout << "ВНИМАНИЕ: достигнут лимит итераций (max_iter), сходимость не гарантирована.\n";
    }
    std::cout << "Результат записан в " << output_file << std::endl;

    return 0;
}
