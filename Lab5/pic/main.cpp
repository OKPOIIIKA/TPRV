#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

const char* kernelSource = R"kernel(
__kernel void modifyChannels(__global const uchar* src, __global uchar* blueChannel,
                             __global uchar* yellowChannel, int rows, int cols) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        uchar red = src[idx * 3 + 2];
        uchar green = src[idx * 3 + 1];
        uchar blue = src[idx * 3];

        blueChannel[idx] = blue - (green + blue) / 2;
        yellowChannel[idx] = red + green - 2 * (abs(red - green) + blue);
    }
}
)kernel";

int changeImage(cv::Mat& src, std::string& res)
{
    auto start = std::chrono::high_resolution_clock::now();
    //выведение инфы о всех платформах и выбор первой платформы
    //предсталяющую GPU
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    auto platform = platforms.front();
    //устройства на выбранной платформе, сохранение в devices
    //и выбор девайса
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    auto device = devices.front();
    //создание контекста, окружение выполнения для ядра 
    //создание объекта с использ. контекста и ядра
    cl::Context context(device);
    cl::Program program(context, kernelSource);
    program.build("-cl-std=CL1.2");
    //очередь команд для выполнения задач на выбранном устройстве и в созданном контексте. 
    //отправка задач на выполнение на устройстве.
    cl::CommandQueue queue(context, device);

    // Создание буферов
    cl::Buffer clSrc(context, CL_MEM_READ_ONLY, src.total() * src.elemSize());
    cl::Buffer clBlueChannel(context, CL_MEM_WRITE_ONLY, src.total());
    cl::Buffer clYellowChannel(context, CL_MEM_WRITE_ONLY, src.total());

    // Копирование данных в буфер
    queue.enqueueWriteBuffer(clSrc, CL_TRUE, 0, src.total() * src.elemSize(), src.data);

    // Создание объекта ядра, установка аргументов и запуск ядра
    cl::Kernel kernel(program, "modifyChannels");
    kernel.setArg(0, clSrc);
    kernel.setArg(1, clBlueChannel);
    kernel.setArg(2, clYellowChannel);
    kernel.setArg(3, src.rows);
    kernel.setArg(4, src.cols);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(src.cols, src.rows));
    queue.finish();
    
    // Получение результата
    cv::Mat blueChannel(src.size(), CV_8UC1);
    cv::Mat yellowChannel(src.size(), CV_8UC1);
    queue.enqueueReadBuffer(clBlueChannel, CL_TRUE, 0, src.total(), blueChannel.data);
    queue.enqueueReadBuffer(clYellowChannel, CL_TRUE, 0, src.total(), yellowChannel.data);

    // Сохранение изображений
    cv::imwrite(res + "_blue_channel.jpg", blueChannel);
    cv::imwrite(res + "_yellow_channel.jpg", yellowChannel);
    
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    return duration;
}

int main() 
{
    std::string output = "1";

    cv::Mat src = cv::imread("img1.jpg");
    if (src.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    auto start = std::chrono::steady_clock::now();
    std::cout << "Picture 1\n";
    for (int i = 0; i < 10; ++i)
    {
        int duration = changeImage(src, output);
        std::cout << duration << " ms\n";
    }

    std::cout << "\n";

    output = "2";

    src = cv::imread("img2.jpg");
    if (src.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    start = std::chrono::steady_clock::now();
    std::cout << "Picture 2\n";
    for (int i = 0; i < 10; ++i)
    {
        int duration = changeImage(src, output);
        std::cout << duration << " ms\n";
    }

    std::cout << "\n";

    output = "3";

    src = cv::imread("img3.jpg");
    if (src.empty()) {
        std::cerr << "Error loading the image" << std::endl;
        return -1;
    }

    start = std::chrono::steady_clock::now();
    std::cout << "Picture 3\n";
    for (int i = 0; i < 10; ++i)
    {
        int duration = changeImage(src, output);
        std::cout << duration << " ms\n";
    }

}