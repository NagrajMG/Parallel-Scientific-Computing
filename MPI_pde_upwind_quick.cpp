#include <mpi.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

namespace {

double initial_profile(double x) {
    double pi = acos(-1.0);
    if (x >= 0.0 && x <= 0.5) return sin(4.0 * pi * x);
    return 0.0;
}

double exact_solution(double x, double t, double c) {
    return initial_profile(x - c * t);
}

// decompostiion for ranks
void build_counts_displs(int n_global, int n_ranks, vector<int>& counts, vector<int>& displs) {
    counts.assign(n_ranks, 0);
    displs.assign(n_ranks, 0);

    int base = n_global / n_ranks;
    int rem = n_global % n_ranks;

    int offset = 0;
    for (int r = 0; r < n_ranks; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

filesystem::path output_dir() {
    if (filesystem::exists("q1") && filesystem::is_directory("q1")) {
        return filesystem::path("q1") / "output";
    }
    return filesystem::path("output");
}

void exchange_halos(
    vector<double>& u,
    int local_n,
    int halo,
    int left_rank,
    int right_rank,
    int left_n,
    int right_n,
    MPI_Comm comm
) {
    // clean up the halo
    for (int h = 0; h < halo; ++h) {
        u[h] = 0.0;
        u[halo + local_n + h] = 0.0;
    }

    int send_left = min(halo, local_n);
    int send_right = min(halo, local_n);
    int recv_left = min(halo, left_n);
    int recv_right = min(halo, right_n);

    MPI_Sendrecv(
        u.data() + halo,
        send_left,
        MPI_DOUBLE,
        left_rank,
        100,
        u.data() + halo + local_n,
        recv_right,
        MPI_DOUBLE,
        right_rank,
        100,
        comm,
        MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        u.data() + halo + local_n - send_right,
        send_right,
        MPI_DOUBLE,
        right_rank,
        101,
        u.data() + halo - recv_left,
        recv_left,
        MPI_DOUBLE,
        left_rank,
        101,
        comm,
        MPI_STATUS_IGNORE
    );
}

void gather_snapshot(
    const vector<double>& local_u,
    int local_n,
    int halo,
    vector<int>& counts,
    vector<int>& displs,
    int rank,
    double* recvbuf,
    MPI_Comm comm
) {
    MPI_Gatherv(
        local_u.data() + halo,
        local_n,
        MPI_DOUBLE,
        recvbuf,
        rank == 0 ? counts.data() : nullptr,
        rank == 0 ? displs.data() : nullptr,
        MPI_DOUBLE,
        0,
        comm
    );
}

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    double c = 1.0;
    double L = 2.0;
    double dx = 0.002;
    double dt = 0.0001;
    double t_end = 1.0;
    int halo = 2;

    const int nx = lround(L / dx) + 1;
    const int nsteps = lround(t_end / dt);

    vector<int> counts, displs;
    build_counts_displs(nx, n_ranks, counts, displs);

    int local_n = counts[rank];
    int start = displs[rank];
    int end = start + local_n - 1;

    int left_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int right_rank = (rank == n_ranks - 1) ? MPI_PROC_NULL : rank + 1;
    int left_n = (left_rank == MPI_PROC_NULL) ? 0 : counts[left_rank];
    int right_n = (right_rank == MPI_PROC_NULL) ? 0 : counts[right_rank];

    vector<double> upwind(local_n + 2 * halo, 0.0), upwind_new(local_n + 2 * halo, 0.0);
    vector<double> quick(local_n + 2 * halo, 0.0), quick_new(local_n + 2 * halo, 0.0);

    for (int li = 0; li < local_n; ++li) {
        int gi = start + li;
        double x = gi * dx;
        double u0 = initial_profile(x);
        upwind[li + halo] = u0;
        quick[li + halo] = u0;
    }

    if (local_n > 0) {
        if (start == 0) {
            upwind[halo] = 0.0;
            quick[halo] = 0.0;
        }
        if (end == nx - 1) {
            upwind[halo + local_n - 1] = 0.0;
            quick[halo + local_n - 1] = 0.0;
        }
    }

    vector<int> snapshot_steps = {0, nsteps / 2, nsteps};
    vector<double> snapshot_times = {0.0, 0.5, 1.0};

    vector<vector<double>> upwind_snap;
    vector<vector<double>> quick_snap;
    if (rank == 0) {
        upwind_snap.assign(3, vector<double>(nx, 0.0));
        quick_snap.assign(3, vector<double>(nx, 0.0));
    }

    auto save_snapshot = [&](size_t snap_id) {
        gather_snapshot(
            upwind,
            local_n,
            halo,
            counts,
            displs,
            rank,
            rank == 0 ? upwind_snap[snap_id].data() : nullptr, // only rank 0 will provide a receive buffer
            MPI_COMM_WORLD
        );
        gather_snapshot(
            quick,
            local_n,
            halo,
            counts,
            displs,
            rank,
            rank == 0 ? quick_snap[snap_id].data() : nullptr, // only rank 0 will provide a receive buffer
            MPI_COMM_WORLD
        );
    };

    size_t snap_id = 0;
    if (snapshot_steps[snap_id] == 0) {
        save_snapshot(snap_id);
        ++snap_id;
    }

    for (int step = 1; step <= nsteps; ++step) {
        if (local_n > 0) {
            if (start == 0) {
                upwind[halo] = 0.0;
                quick[halo] = 0.0;
            }
            if (end == nx - 1) {
                upwind[halo + local_n - 1] = 0.0;
                quick[halo + local_n - 1] = 0.0;
            }
        }

        exchange_halos(upwind, local_n, halo, left_rank, right_rank, left_n, right_n, MPI_COMM_WORLD);
        exchange_halos(quick, local_n, halo, left_rank, right_rank, left_n, right_n, MPI_COMM_WORLD);

        for (int li = 0; li < local_n; ++li) {
            int gi = start + li;
            int idx = li + halo;

            if (gi == 0 || gi == nx - 1) 
            {
                upwind_new[idx] = 0.0;
            } 
            else 
            {
                double dudx_up = (upwind[idx] - upwind[idx - 1]) / dx;
                upwind_new[idx] = upwind[idx] - c * dt * dudx_up;
            }

            if (gi == 0 || gi == nx - 1) 
            {
                quick_new[idx] = 0.0;
            } 
            else if (gi == 1) {
            
                double dudx_up = (quick[idx] - quick[idx - 1]) / dx;
                quick_new[idx] = quick[idx] - c * dt * dudx_up;
            } 
            else 
            {
                double dudx_quick = (
                    (3.0 / 8.0) * quick[idx]
                    - (7.0 / 8.0) * quick[idx - 1]
                    + (1.0 / 8.0) * quick[idx - 2]
                    + (3.0 / 8.0) * quick[idx + 1]
                ) / dx;
                quick_new[idx] = quick[idx] - c * dt * dudx_quick;
            }
        }

        upwind.swap(upwind_new);
        quick.swap(quick_new);

        if (snap_id < snapshot_steps.size() && step == snapshot_steps[snap_id]) {
            save_snapshot(snap_id);
            ++snap_id;
        }
    }

    // output is written by the rank 0 root
    if (rank == 0) {
        vector<double> x(nx, 0.0);
        for (int i = 0; i < nx; ++i) {
            x[i] = i * dx;
        }

        vector<vector<double>> exact_snap(3, vector<double>(nx, 0.0));
        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < nx; ++i) {
                exact_snap[k][i] = exact_solution(x[i], snapshot_times[k], c);
            }
        }

        auto out_dir = output_dir();
        filesystem::create_directories(out_dir);
        auto out_file = out_dir / ("q1_mpi_p" + to_string(n_ranks) + ".csv");

        ofstream ofs(out_file);
        if (!ofs) {
            cerr << "Failed to open output file: " << out_file << "\n";
            MPI_Finalize();
            return 1;
        }

        ofs << setprecision(10);
        ofs << "x,exact_t0,upwind_t0,quick_t0,"
            << "exact_t05,upwind_t05,quick_t05,"
            << "exact_t10,upwind_t10,quick_t10\n";

        for (int i = 0; i < nx; ++i) {
            ofs << x[i] << ","
                << exact_snap[0][i] << "," << upwind_snap[0][i] << "," << quick_snap[0][i] << ","
                << exact_snap[1][i] << "," << upwind_snap[1][i] << "," << quick_snap[1][i] << ","
                << exact_snap[2][i] << "," << upwind_snap[2][i] << "," << quick_snap[2][i]
                << "\n";
        }

    }

    MPI_Finalize();
    return 0;
}
