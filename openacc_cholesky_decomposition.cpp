#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#define TYPE float
#define SMALLVALUE 0.001f

static inline double now_seconds()
{
    using clock = std::chrono::high_resolution_clock;
    static const auto start = clock::now();
    const auto t = clock::now();
    return std::chrono::duration<double>(t - start).count();
}

void initmult(TYPE* mat, int N)
{
    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            mat[ii * N + jj] = 0.0f;
        }
    }

    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < ii; ++jj) {
            mat[ii * N + jj] = (ii + jj) / (TYPE)N / N;
            mat[jj * N + ii] = (ii + jj) / (TYPE)N / N;
        }
    }

    for (int ii = 0; ii < N; ++ii) {
        mat[ii * N + ii] = 1.0f;
    }
}

void copyMat(TYPE* dst, const TYPE* src, int N)
{
    for (int idx = 0; idx < N * N; ++idx) {
        dst[idx] = src[idx];
    }
}

void zeroUpperTriangle(TYPE* a, int N)
{
    for (int ii = 0; ii < N; ++ii) {
        for (int jj = ii + 1; jj < N; ++jj) {
            a[ii * N + jj] = 0.0f;
        }
    }
}

void printMat(const TYPE* a, int N)
{
    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            std::printf("%9.4f ", a[ii * N + jj]);
        }
        std::printf("\n");
    }
}

void cholesky_serial(TYPE* a, int N)
{
    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < ii; ++jj) {
            for (int kk = 0; kk < jj; ++kk) {
                a[ii * N + jj] += -a[ii * N + kk] * a[jj * N + kk];
            }

            a[ii * N + jj] /= (a[jj * N + jj] > SMALLVALUE ? a[jj * N + jj] : 1.0f);
        }

        for (int kk = 0; kk < ii; ++kk) {
            a[ii * N + ii] += -a[ii * N + kk] * a[ii * N + kk];
        }

        if (a[ii * N + ii] < 0.0f && a[ii * N + ii] > -1.0e-5f) {
            a[ii * N + ii] = 0.0f;
        }

        a[ii * N + ii] = sqrtf(a[ii * N + ii]);
    }

    zeroUpperTriangle(a, N);
}

void cholesky_openacc(TYPE* a, int N, int gangs, int vector_length)
{
    const int total = N * N;
    int ok = 1;

#pragma acc data copy(a[0:total]) copy(ok)
    {
        for (int jj = 0; jj < N; ++jj) {

#pragma acc serial present(a, ok)
            {
                TYPE diag_sum = 0.0f;

                for (int kk = 0; kk < jj; ++kk) {
                    diag_sum += a[jj * N + kk] * a[jj * N + kk];
                }

                a[jj * N + jj] -= diag_sum;

                if (a[jj * N + jj] < 0.0f && a[jj * N + jj] > -1.0e-5f) {
                    a[jj * N + jj] = 0.0f;
                }

                if (a[jj * N + jj] <= 0.0f) {
                    ok = 0;
                    a[jj * N + jj] = 0.0f;
                } else {
                    a[jj * N + jj] = sqrtf(a[jj * N + jj]);
                }
            }

#pragma acc parallel loop present(a, ok) num_gangs(gangs) vector_length(vector_length)
            for (int ii = jj + 1; ii < N; ++ii) {
                if (ok) {
                    TYPE sum = 0.0f;

#pragma acc loop seq
                    for (int kk = 0; kk < jj; ++kk) {
                        sum += a[ii * N + kk] * a[jj * N + kk];
                    }

                    a[ii * N + jj] =
                        (a[ii * N + jj] - sum) /
                        (a[jj * N + jj] > SMALLVALUE ? a[jj * N + jj] : 1.0f);
                }
            }
        }

#pragma acc parallel loop present(a)
        for (int ii = 0; ii < N; ++ii) {
            for (int jj = ii + 1; jj < N; ++jj) {
                a[ii * N + jj] = 0.0f;
            }
        }
    }
}

TYPE maxAbsDiff(const TYPE* a, const TYPE* b, int N)
{
    TYPE maxdiff = 0.0f;

    for (int idx = 0; idx < N * N; ++idx) {
        TYPE diff = fabsf(a[idx] - b[idx]);
        if (diff > maxdiff) {
            maxdiff = diff;
        }
    }

    return maxdiff;
}

TYPE reconstructionError(const TYPE* original, const TYPE* L, int N)
{
    TYPE maxerr = 0.0f;

    for (int ii = 0; ii < N; ++ii) {
        for (int jj = 0; jj < N; ++jj) {
            TYPE val = 0.0f;
            int lim = (ii < jj) ? ii : jj;

            for (int kk = 0; kk <= lim; ++kk) {
                val += L[ii * N + kk] * L[jj * N + kk];
            }

            TYPE err = fabsf(original[ii * N + jj] - val);
            if (err > maxerr) {
                maxerr = err;
            }
        }
    }

    return maxerr;
}

void run_case(int N, int gangs, int vector_length, std::ofstream& csv)
{
    TYPE* original = new TYPE[N * N];
    TYPE* serial_mat = new TYPE[N * N];
    TYPE* acc_mat = new TYPE[N * N];

    initmult(original, N);
    copyMat(serial_mat, original, N);
    copyMat(acc_mat, original, N);

    double t1 = now_seconds();
    cholesky_serial(serial_mat, N);
    double t2 = now_seconds();

    double t3 = now_seconds();
    cholesky_openacc(acc_mat, N, gangs, vector_length);
    double t4 = now_seconds();

    TYPE diff = maxAbsDiff(serial_mat, acc_mat, N);

    TYPE serial_rec_err = -1.0f;
    TYPE acc_rec_err = -1.0f;

    if (N <= 10) {
        serial_rec_err = reconstructionError(original, serial_mat, N);
        acc_rec_err = reconstructionError(original, acc_mat, N);
    }

    std::printf("\nN = %d\n", N);
    std::printf("Serial time  = %.6f seconds\n", t2 - t1);
    std::printf("OpenACC time = %.6f seconds\n", t4 - t3);
    std::printf("Max absolute difference = %.8e\n", diff);

    if (N <= 10) {
        std::printf("Serial reconstruction error  = %.8e\n", serial_rec_err);
        std::printf("OpenACC reconstruction error = %.8e\n", acc_rec_err);
    }

    csv << N << ","
        << std::setprecision(12) << (t2 - t1) << ","
        << std::setprecision(12) << (t4 - t3) << ","
        << std::setprecision(12) << diff << ","
        << std::setprecision(12) << serial_rec_err << ","
        << std::setprecision(12) << acc_rec_err << ","
        << gangs << ","
        << vector_length << "\n";

    if (N == 10) {
        std::printf("\nSerial lower triangular matrix L:\n");
        printMat(serial_mat, N);

        std::printf("\nOpenACC lower triangular matrix L:\n");
        printMat(acc_mat, N);
    }

    delete[] original;
    delete[] serial_mat;
    delete[] acc_mat;
}

int main(int argc, char** argv)
{
    int gangs = 128;
    int vector_length = 128;

    if (argc >= 2) {
        gangs = std::atoi(argv[1]);
    }

    if (argc >= 3) {
        vector_length = std::atoi(argv[2]);
    }

    std::ofstream csv("timings.csv");
    csv << "N,serial_seconds,openacc_seconds,max_abs_diff,"
           "serial_reconstruction_error,openacc_reconstruction_error,"
           "gangs,vector_length\n";

    std::printf("Cholesky decomposition using Serial C++ and OpenACC\n");
    std::printf("gangs = %d, vector_length = %d\n", gangs, vector_length);

    run_case(10, gangs, vector_length, csv);
    run_case(100, gangs, vector_length, csv);
    run_case(1000, gangs, vector_length, csv);

    std::printf("\nTiming results written to timings.csv\n");

    return 0;
}