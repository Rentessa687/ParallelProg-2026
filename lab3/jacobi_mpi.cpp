#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Использование: mpirun -np P " << argv[0]
                      << " <input_file> [output_file] [eps] [max_iter]" << std::endl;
        MPI_Finalize();
        return 1;
    }
    const std::string input_file = argv[1];
    const std::string output_file = (argc >= 3) ? argv[2] : "output.txt";
    const double eps = (argc >= 4) ? std::atof(argv[3]) : 1e-6;
    const int max_iter = (argc >= 5) ? std::atoi(argv[4]) : 100000;

    int n = 0;
    std::vector<double> A, b;
    int ok = 1;
    if (rank == 0) {
        if (!readInput(input_file, n, A, b)) ok = 0;
        else {
            for (int i = 0; i < n; ++i) {
                if (std::fabs(A[(size_t)i * n + i]) < 1e-14) {
                    std::cerr << "Ошибка: нулевой диагональный элемент A[" << i << "][" << i << "]" << std::endl;
                    ok = 0;
                    break;
                }
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        MPI_Finalize();
        return 1;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> rows(size), row_displs(size);
    for (int p = 0, off = 0; p < size; ++p) {
        rows[p] = n / size + (p < n % size ? 1 : 0);
        row_displs[p] = off;
        off += rows[p];
    }
    const int local_n = rows[rank];
    const int first_row = row_displs[rank];

    std::vector<int> a_counts(size), a_displs(size);
    for (int p = 0; p < size; ++p) {
        a_counts[p] = rows[p] * n;
        a_displs[p] = row_displs[p] * n;
    }

    std::vector<double> A_loc((size_t)local_n * n), b_loc(local_n);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_scatter0 = MPI_Wtime();
    MPI_Scatterv(A.data(), a_counts.data(), a_displs.data(), MPI_DOUBLE,
                 A_loc.data(), local_n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b.data(), rows.data(), row_displs.data(), MPI_DOUBLE,
                 b_loc.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const double scatter_ms = (MPI_Wtime() - t_scatter0) * 1000.0;
    std::vector<double>().swap(A);

    std::vector<double> x_buf(n, 0.0), x_new_buf(n, 0.0);
    double *x = x_buf.data();
    double *x_new = x_new_buf.data();
    std::vector<double> x_loc(local_n);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    int iter = 0;
    double diff = 0.0;
    for (iter = 0; iter < max_iter; ++iter) {
        double diff_sq = 0.0;
        for (int li = 0; li < local_n; ++li) {
            const int gi = first_row + li;
            const double *Ai = A_loc.data() + (size_t)li * n;
            double sigma = 0.0;
            for (int j = 0; j < gi; ++j) sigma += Ai[j] * x[j];
            for (int j = gi + 1; j < n; ++j) sigma += Ai[j] * x[j];
            const double xi = (b_loc[li] - sigma) / Ai[gi];
            x_loc[li] = xi;
            const double d = xi - x[gi];
            diff_sq += d * d;
        }

        MPI_Allgatherv(x_loc.data(), local_n, MPI_DOUBLE,
                       x_new, rows.data(), row_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        double diff_sq_global = 0.0;
        MPI_Allreduce(&diff_sq, &diff_sq_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        diff = std::sqrt(diff_sq_global);
        std::swap(x, x_new);

        if (diff < eps) {
            ++iter;
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed_ms = (MPI_Wtime() - t0) * 1000.0;
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed_ms, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double res_loc = 0.0;
    for (int li = 0; li < local_n; ++li) {
        const double *Ai = A_loc.data() + (size_t)li * n;
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += Ai[j] * x[j];
        const double r = s - b_loc[li];
        res_loc += r * r;
    }
    double res_glob = 0.0;
    MPI_Reduce(&res_loc, &res_glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int rc = 0;
    if (rank == 0) {
        const double residual = std::sqrt(res_glob);
        std::ofstream fout(output_file);
        if (!fout) {
            std::cerr << "Ошибка: не удалось открыть выходной файл " << output_file << std::endl;
            rc = 1;
        } else {
            fout.precision(12);
            fout << n << "\n";
            for (int i = 0; i < n; ++i) fout << x[i] << (i + 1 < n ? ' ' : '\n');
            fout << "iterations " << iter << "\n";
            fout << "time_ms " << elapsed_max << "\n";
            fout << "residual " << residual << "\n";
            fout << "eps " << eps << "\n";
            fout << "problem_size " << n << "\n";
            fout << "processes " << size << "\n";
            fout << "scatter_ms " << scatter_ms << "\n";

            std::cout << "Метод Якоби (MPI) завершён.\n";
            std::cout << "n = " << n << ", процессов = " << size
                      << ", итераций = " << iter << ", время = " << elapsed_max << " мс\n";
            std::cout << "невязка ||Ax-b|| = " << residual << "\n";
            if (diff >= eps && iter >= max_iter)
                std::cout << "ВНИМАНИЕ: достигнут лимит итераций (max_iter), сходимость не гарантирована.\n";
            std::cout << "Результат записан в " << output_file << std::endl;
        }
    }

    MPI_Finalize();
    return rc;
}
