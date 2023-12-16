#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Ядро CUDA для преобразования в полутоновое изображение и масштабирования
__global__ void processImageGPU(const uchar* input, uchar* output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row * 2 < height * 2 && col * 2 < width * 2) {
        int index_in = row * width + col;
        int index_out = (row * 2) * (width * 2) + (col * 2);

        uchar intensity = (input[index_in * 3] + input[index_in * 3 + 1] + input[index_in * 3 + 2]) / 3;

        output[index_out] = intensity;
        output[index_out + 1] = intensity;
        output[index_out + width * 2] = intensity;
        output[index_out + width * 2 + 1] = intensity;
    }
}
void processImageCPU(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;

    output = cv::Mat(height * 2, width * 2, CV_8UC1);

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            uchar intensity = (input.at<cv::Vec3b>(row, col)[0] +
                               input.at<cv::Vec3b>(row, col)[1] +
                               input.at<cv::Vec3b>(row, col)[2]) / 3;

            output.at<uchar>(row * 2, col * 2) = intensity;
            output.at<uchar>(row * 2, col * 2 + 1) = intensity;
            output.at<uchar>(row * 2 + 1, col * 2) = intensity;
            output.at<uchar>(row * 2 + 1, col * 2 + 1) = intensity;
        }
    }

    // Сохранение результата
    cv::imwrite("../res/CPU3.png", output);
}

int main() {
    // Загрузка изображения
    cv::Mat img = cv::imread("../img/img3.jpg");

    if (img.empty()) {
        std::cerr << "Не удалось загрузить изображение.\n";
        return -1;
    }

    // Переменные для замера времени
    cv::TickMeter tm;
    
    // Замеряем время выполнения на CPU
    tm.start();
    for (int i = 0; i < 10; ++i) {
        // Преобразование в полутоновое изображение и масштабирование
        cv::Mat resultCPU;
        processImageCPU(img, resultCPU);
    }
    tm.stop();
    std::cout << "Время выполнения на CPU (10 итераций): " << tm.getTimeMilli() << " мс\n";

    int width = img.cols;
    int height = img.rows;

    uchar *h_input = img.data;
    uchar *h_output = new uchar[width * 2 * height * 2];
    uchar *d_input, *d_output;

    cudaMalloc(&d_input, sizeof(uchar) * width * height * 3);
    cudaMalloc(&d_output, sizeof(uchar) * width * 2 * height * 2);

    cudaMemcpy(d_input, h_input, sizeof(uchar) * width * height * 3, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Замеряем время выполнения на GPU
    cv::TickMeter tm1;
    tm1.start();

    for (int i = 0; i < 10; ++i) {
        // Преобразование в полутоновое изображение и масштабирование
        processImageGPU<<<grid, block>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();
    }

    tm1.stop();
    std::cout << "Время выполнения на GPU (10 итераций): " << tm1.getTimeMilli() << " мс\n";

    cudaMemcpy(h_output, d_output, sizeof(uchar) * width * 2 * height * 2, cudaMemcpyDeviceToHost);

    // Преобразование результата в матрицу OpenCV
    cv::Mat resultGPU(height * 2, width * 2, CV_8UC1, h_output);

    // Сохранение результата
    cv::imwrite("../res/GPU3.png", resultGPU);

    // Освобождение памяти
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    return 0;
}