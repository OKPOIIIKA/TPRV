#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <thread>

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

int sse_mul(double** A, double** B, double** C, int N)
{
    cout << "SSE mul:" << endl;
    double** B1 = new double* [N];
    for (int i = 0; i < N; i++)
    {
        B1[i] = new double[N];
        for (int j = 0; j < N; j++)
        {
            B1[i][j] = B[j][i];
        }
    }
    auto start = high_resolution_clock::now();
    
    if (N < 2) return 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            __m128d add = _mm_setzero_pd(); // заполняем каждый бит в результирующем SSE-регистре нулями
            for (int k = 0; k < N; k += 2)
            {
                __m128d a_line = _mm_loadu_pd(&A[i][k]); // загружаем 2 элемента double из массива A в невыровненный SSE-регистр 
                __m128d b_line = _mm_loadu_pd(&B1[j][k]); // загружаем 2 элемента double из массива B в невыровненный SSE-регистр
                __m128d mul = _mm_mul_pd(a_line, b_line); // умножение двух SSE-регистров
                add = _mm_add_pd(mul, add); //складывание умножения для добавления в результирующую матрицу
            }
            __m128d c_line = _mm_hadd_pd(add, add);
            _mm_storeu_pd(&C[i][j], c_line);
        }
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "time sse: " << duration.count() << " milliseconds" << endl;
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


void scalar_thread(double** A, double** B, double** C, int N, int start, int end)
{
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


int sse_thread_mul(double** A, double** B, double** C, int N, int start, int end)
{
    double** B1 = new double* [N];
    for (int i = 0; i < N; i++)
    {
        B1[i] = new double[N];
        for (int j = 0; j < N; j++)
        {
            B1[i][j] = B[j][i];
        }
    }

    if (N < 2) return 0;
    for (int i = start; i < end; i++)
    {
        for (int j = 0; j < N; j++)
        {
            __m128d add = _mm_setzero_pd(); // заполняем каждый бит в результирующем SSE-регистре нулями
            for (int k = 0; k < N; k += 2)
            {
                __m128d a_line = _mm_loadu_pd(&A[i][k]); // загружаем 2 элемента double из массива A в невыровненный SSE-регистр 
                __m128d b_line = _mm_loadu_pd(&B1[j][k]); // загружаем 2 элемента double из массива B в невыровненный SSE-регистр
                __m128d mul = _mm_mul_pd(a_line, b_line); // умножение двух SSE-регистров
                add = _mm_add_pd(mul, add); //складывание умножения для добавления в результирующую матрицу
            }
            __m128d c_line = _mm_hadd_pd(add, add);
            _mm_storeu_pd(&C[i][j], c_line);
        }
    }
}

//вариант 2 SSE2 (128 бит) с плавающей точкой, двойной точностью
int main()
{
    const unsigned int MAXTHREADS = 16;
    const unsigned int N = 2048; // размер массива
    double** A = new double* [N]; // выделение памяти под массивы
    double** B = new double* [N]; // выделение памяти под массивы
    double** Scalar = new double* [N]; // выделение памяти под массивы
    double** SSE = new double* [N]; // выделение памяти под массивы
    double** Thread = new double* [N]; // выделение памяти под массивы
    double** ThreadSSE = new double* [N]; // выделение памяти под массивы

    srand(time(NULL));
    for (unsigned int i = 0; i < N; ++i)
    {
        A[i] = new double[N]; // выделение памяти под массивы
        B[i] = new double[N]; // выделение памяти под массивы
        Scalar[i] = new double[N]; // выделение памяти под массивы
        SSE[i] = new double[N]; // выделение памяти под массивы
        Thread[i] = new double[N]; // выделение памяти под массивы
        ThreadSSE[i] = new double[N]; // выделение памяти под массивы
        for (unsigned int j = 0; j < N; ++j)
        {
            A[i][j] = (double)(rand() % 10000) / 100; // заполнение массива
            B[i][j] = (double)(rand() % 10000) / 100; // заполнение массива
            Scalar[i][j] = 0; // заполнение массива
            SSE[i][j] = 0; // заполнение массива
            Thread[i][j] = 0; // заполнение массива
            ThreadSSE[i][j] = 0; // заполнение массива
        }
    }

    scalar_mul(A, B, Scalar, N);

    sse_mul(A, B, SSE, N);
    

    cout << "Threads mul:\n";


    thread threads[MAXTHREADS];

    auto start2 = high_resolution_clock::now();

    for (int i = 0; i < MAXTHREADS; i++) 
    {
        if (i == 0) 
        {
            threads[i] = thread(scalar_thread,A,B, Thread, N, 0, N/MAXTHREADS);
        }
        else 
        {
            threads[i] = thread(scalar_thread, A, B, Thread, N, ((N * i) / MAXTHREADS), ((i + 1) * N) / MAXTHREADS);
        }
    }

    for (auto i = 0; i < MAXTHREADS; i++) {
        threads[i].join();
    }

    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2 - start2);

    cout << "time threads: " << duration2.count() << " milliseconds" << endl;


    cout << "Threads SSE mul:\n";
    thread threadss[MAXTHREADS];

    auto start3 = high_resolution_clock::now();

    for (int i = 0; i < MAXTHREADS; i++)
    {
        if (i == 0)
        {
            threadss[i] = thread(sse_thread_mul, A, B, ThreadSSE, N, 0, N / MAXTHREADS);
        }
        else
        {
            threadss[i] = thread(sse_thread_mul, A, B, ThreadSSE, N, ((N * i) / MAXTHREADS), ((i + 1) * N) / MAXTHREADS);
        }
    }

    for (auto i = 0; i < MAXTHREADS; i++) {
        threadss[i].join();
    }

    auto end3 = high_resolution_clock::now();
    auto duration3 = duration_cast<milliseconds>(end3 - start3);

    cout << "time sse thread: " << duration3.count() << " milliseconds" << endl;
    
    delete[] A;
    delete[] B;
    delete[] Scalar;
    delete[] SSE;
    delete[] Thread;
    delete[] ThreadSSE;
}