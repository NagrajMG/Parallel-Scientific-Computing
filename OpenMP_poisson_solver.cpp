#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <iomanip>
using namespace std;

inline int idx(int i,int j,int N){ 
    return i*N + j; 
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        cout<<"Add delta and threads\n";
        return 1;
    }

    double delta = atof(argv[1]);
    int threads = atoi(argv[2]);

    omp_set_num_threads(threads);

    const int N = int(2.0/delta) + 1;

    cout<<"Grid spacing = "<<delta<<endl;
    cout<<"Grid size = "<<N<<" x "<<N<<endl;
    cout<<"Threads = "<<threads<<endl;

    vector<double> phi_old(N*N,0.0);
    vector<double> phi_new(N*N,0.0);
    vector<double> q(N*N);
    vector<double> exact(N*N);
    vector<double> x(N), y(N);

    // axis
    for(int i=0;i<N;i++)
    {
        x[i] = -1.0 + i*delta;
        y[i] = -1.0 + i*delta;
    }

    #pragma omp parallel for collapse(2) schedule(static) // not unusual overheads for some processes
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            q[idx(i,j,N)] = 2*(2 - x[i]*x[i] - y[j]*y[j]);
            exact[idx(i,j,N)] = (x[i]*x[i]-1)*(y[j]*y[j]-1);
        }
    }

    int iterations = 0;
    double error = 1.0;

    double start = omp_get_wtime();

    while(error > 0.01)
    {
        double num = 0.0;
        double den = 0.0;

        #pragma omp parallel for collapse(2) reduction(+:num,den) schedule(static)
        for(int i=1;i<N-1;i++)
        {
            for(int j=1;j<N-1;j++)
            {
                int id = idx(i,j,N);
                phi_new[id] = 0.25 * (phi_old[idx(i+1,j,N)] + phi_old[idx(i-1,j,N)] + phi_old[idx(i,j+1,N)] + phi_old[idx(i,j-1,N)]) + (delta*delta/4.0) * q[id];
                double diff = phi_new[id] - exact[id];
                num += diff*diff; 
                den += exact[id]*exact[id];
            }
        }
        error = sqrt(num/den);
        swap(phi_old,phi_new);
        iterations++;
    }

    double end = omp_get_wtime();
    double runtime = end-start; 

    cout<<"Iterations = "<<iterations<<endl;
    cout<<"Final error = "<<error<<endl;
    cout<<"Time = "<<runtime << setprecision(8) << " seconds"<<endl;

    //getting the index for y = 0.5
    int j_mid = int((0.5+1.0)/delta + 0.5);

    ofstream file;

    switch(N)
    {
        case 21:
            file.open("output/data/phi_line_parallel_21.dat");
            cout<<"Output file @phi_line_parallel_21.dat\n";
            break;

        case 201:
            file.open("output/data/phi_line_parallel_201.dat");
            cout<<"Output file @phi_line_parallel_201.dat\n";
            break;

        case 401:
            file.open("output/data/phi_line_parallel_401.dat");
            cout<<"Output file @phi_line_parallel_401.dat\n";
            break;

        default:
            cout<<"send correct delta\n";
            return 1;
    }

    for(int i=0;i<N;i++)
    {
        file<<x[i]<<" "<<phi_old[idx(i,j_mid,N)]<<" "<<exact[idx(i,j_mid,N)]<<"\n";
    }

    file.close();
    ofstream timing("output/timing/timing_parallel_delta.dat", ios::app);
    timing<<delta<<" "<<threads<<" "<<runtime << setprecision(8) << "\n";
    timing.close();

    if(delta == 0.005)
    {
        ofstream scaling("output/timing/thread_scaling.dat", ios::app);
        scaling<<threads<<" "<<runtime  << setprecision(8) <<"\n";
        scaling.close();
    }

    return 0;
}