#include <iostream>
#include "mpi.h"
#include <chrono>

int size = 2049;
void print_matrix(double_t* matrix) 
{
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}
double_t* generateMatrix(bool empty) 
{
    double_t* matrix = new double_t[size * size];
    for (int i = 0; i < size * size; ++i) 
    {
        if (!empty) 
        {
            matrix[i] = (double)(rand() % 100) / 10;
        }
        else 
        {
            matrix[i] = 0;
        }
    }
    return matrix;
}
double_t* matrixMultOneProc(double_t* A, double_t* B) 
{
    double_t* res = new double_t[size * size];
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            res[i * size + j] = 0;
            for (int k = 0; k < size; k++) 
            {
                res[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
    return res;
}
void matrixMultAllProc(double_t* A, double_t* B, double_t* res, int chunks) 
{
    for (int i = 0; i < chunks; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            res[i * size + j] = 0;
            for (int k = 0; k < size; k++) 
            {
                res[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}
bool is_equal_matrix(double_t* a, double_t* b) 
{
    for (int i = 0; i < size * size; ++i) 
    {
        if (a[i] != b[i]) 
        {
            std::cout << i << ": " << a[i] << ' ' << b[i] << '\n';
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank_id, num_proc, chunk, offset;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    double_t* A = nullptr;
    double_t* B = nullptr;
    double_t* AllProc = nullptr;
    double_t* OneProc = nullptr;
    chunk = size / (num_proc - 1);
    if (rank_id == 0) 
    {
        A = generateMatrix(false);
        B = generateMatrix(false);
        AllProc = generateMatrix(true);
        OneProc = generateMatrix(true);
        auto start = std::chrono::high_resolution_clock::now();
        offset = 0;
        for (int dest = 1; dest <= num_proc - 1; dest++) 
        {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&A[offset * size], chunk * size, MPI_INT64_T, dest, 1, MPI_COMM_WORLD);
            MPI_Send(B, size * size, MPI_INT64_T, dest, 1, MPI_COMM_WORLD);
            offset += chunk;
        }
        for (int src = 1; src <= num_proc - 1; src++) 
        {
            MPI_Recv(&offset, 1, MPI_INT, src, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&AllProc[offset * size], chunk * size, MPI_INT64_T, src, 2, MPI_COMM_WORLD, &status);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "MPI: " << duration << " milliseconds\n";
        start = std::chrono::high_resolution_clock::now();
        OneProc = matrixMultOneProc(A, B);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Scalar: " << duration <<  " milliseconds\n";
        std::cout << "Matrices are equal: " << is_equal_matrix(AllProc, OneProc) << '\n';
    }
    else 
    {
        int src = 0;
        MPI_Recv(&offset, 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
        A = new double_t[chunk * size];
        MPI_Recv(A, chunk * size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, &status);
        B = new double_t[size * size];
        MPI_Recv(B, size * size, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, &status);

        AllProc = new double_t[chunk * size];
        matrixMultAllProc(A, B, AllProc, chunk);

        MPI_Send(&offset, 1, MPI_INT, src, 2, MPI_COMM_WORLD);
        MPI_Send(AllProc, chunk * size, MPI_DOUBLE, src, 2, MPI_COMM_WORLD);
    }

    delete[] A;
    delete[] B;
    delete[] AllProc;
    delete[] OneProc;

    MPI_Finalize();
    return 0;
}