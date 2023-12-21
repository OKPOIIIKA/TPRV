#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl2.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

const char *kernelSource = R"(
__kernel void processImage(__global const uchar3* inputImage,
                           __global uchar3* outputImage,
                           const int width,
                           const int height) {
    int gidX = get_global_id(0);
    int gidY = get_global_id(1);

    // Проверка выхода за границы изображения
    if (gidX < width && gidY < height) {
        int index = gidY * width + gidX;

        // Получение интенсивности пикселя
        uchar intensity = (inputImage[index].s0 + inputImage[index].s1 + inputImage[index].s2) / 3;

        // Запись интенсивности в результирующий буфер
        outputImage[index].x = intensity;
        outputImage[index].y = intensity;
        outputImage[index].z = intensity;
    }
}
)";
void checkErr(cl_int err, const char *name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {

// Получение доступных платформ
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    // Выбор первой платформы
    cl::Platform platform = platforms.front();
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    // Получение доступных устройств
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        std::cerr << "No OpenCL GPU devices found." << std::endl;
        return 1;
    }

    // Выбор первого устройства
    cl::Device device = devices.front();
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // Получение информации об устройстве
    cl_uint max_compute_units;
    device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &max_compute_units);
    std::cout << "Max Compute Units: " << max_compute_units << std::endl;


    // Загрузка изображения
    cv::Mat originalImage = cv::imread("img1.jpg");
    if (originalImage.empty()) {
        std::cerr << "Error: Unable to load input image!" << std::endl;
        return EXIT_FAILURE;
    }

    int imageWidth = originalImage.cols;
    int imageHeight = originalImage.rows;
    
    for (int i = 0; i < 10; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Инициализация OpenCL
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found" << std::endl;
            return EXIT_FAILURE;
        }   
        //создание контекста (GPU)
        cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Чтение ядра OpenCL из строки
        std::string kernelCode(kernelSource);
        cl::Program::Sources source;
        source.push_back({kernelCode.c_str(), kernelCode.length()});
        cl::Program program(context, source);
        cl_int err = program.build(devices, "");
        checkErr(err, "Program::build()");

        // Создание ядра
        cl::Kernel kernel(program, "processImage");

        // Создание буферов для данных
        cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(uchar) * originalImage.total() * originalImage.channels(), originalImage.data, &err);
        checkErr(err, "Buffer::Buffer()");

        cv::Mat outputImage(originalImage.size(), originalImage.type());

        cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * outputImage.total() * outputImage.channels(), nullptr, &err);
        checkErr(err, "Buffer::Buffer()");

        // Установка аргументов для ядра
        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);
        kernel.setArg(2, imageWidth);
        kernel.setArg(3, imageHeight);

        // Выполнение ядра на GPU
        cl::CommandQueue queue(context, devices[0]);
        cl::Event event;
        //запуск ядра в очередь команд
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight), cl::NullRange, nullptr, &event);
        event.wait();

        // Чтение результатов из буфера вывода в системную память
        queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(uchar) * outputImage.total() * outputImage.channels(), outputImage.data);

        // Масштабирование изображения в два раза
        cv::Mat scaledImage;
        cv::resize(outputImage, scaledImage, cv::Size(), 2, 2);

        // Сохранение результата
        cv::imwrite("output1.jpg", scaledImage);

        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

        std::cout << "Pic 1 " << i + 1 << " duration: " << duration << " ms\n";
    }
    return 0;
}
