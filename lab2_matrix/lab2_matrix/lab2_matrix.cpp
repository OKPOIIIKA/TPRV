#include <iostream>
#include <omp.h>
#include <chrono>


using namespace std;
using namespace std::chrono;

bool Comparison(double** C, double** Scalar, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (C[i][j] != Scalar[i][j]) return false;
        }
    }
    return true;
}

void Show(double** Matrix, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << Matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


void scalar_mul(double** A, double** B, double** C, int N)
{
    auto start1 = high_resolution_clock::now();

    cout << "Scalar mul:" << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start1);

    cout << "time scalar: " << duration1.count() << " milliseconds" << endl;
}

void OpenMP_mul(double** A, double** B, double** C, int N)
{
    auto start1 = high_resolution_clock::now();
    cout << "OpenMP mul:" << endl;
    omp_set_num_threads(4);
    #pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
   
    
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start1);

    cout << "time OpenMP: " << duration1.count() << " milliseconds" << endl;
}

void OpenMP_mul1(double** A, double** B, double** C, int N)
{
    auto start1 = high_resolution_clock::now();
    cout << "OpenMP mul:" << endl;
    //omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start1);

    cout << "time OpenMP: " << duration1.count() << " milliseconds" << endl;
}


int main()
{
    const unsigned int N = 4096; // размер массива
    double** A = new double* [N]; // выделение памяти под массивы
    double** B = new double* [N]; // выделение памяти под массивы
    double** Scalar = new double* [N]; // выделение памяти под массивы
    double** OpenMP = new double* [N]; // выделение памяти под массивы

    srand(time(NULL));
    for (unsigned int i = 0; i < N; ++i)
    {
        A[i] = new double[N]; // выделение памяти под массивы
        B[i] = new double[N]; // выделение памяти под массивы
        Scalar[i] = new double[N]; // выделение памяти под массивы
        OpenMP[i] = new double[N]; // выделение памяти под массивы
        for (unsigned int j = 0; j < N; ++j)
        {
            A[i][j] = (double)(rand() % 10000) / 100; // заполнение массива
            B[i][j] = (double)(rand() % 10000) / 100; // заполнение массива
            Scalar[i][j] = 0; // заполнение массива
            OpenMP[i][j] = 0; // заполнение массива
        }
    }

   // scalar_mul(A, B, Scalar, N);
    OpenMP_mul(A, B, OpenMP, N);
    OpenMP_mul1(A, B, OpenMP, N);

    delete[] A;
    delete[] B;
    delete[] Scalar;
    delete[] OpenMP;

    return 0;
}