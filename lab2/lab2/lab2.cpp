// lab2.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace chrono;

int main() {
    // Открываем изображение с помощью OpenCV
    cv::Mat input_image = cv::imread("C:/input_image2.jpg");

    // Проверяем, что изображение успешно загружено
    if (input_image.empty()) {
        std::cerr << "Не удалось загрузить изображение." << std::endl;
        return -1;
    }
    //// Получаем размеры исходного изображения
    int width = input_image.cols;
    int height = input_image.rows;

    // Создаем новое полутоновое изображение с размерами, увеличенными в 2 раза
    cv::Mat output_image(height * 2, width * 2, CV_8UC1);

    auto start1 = high_resolution_clock::now();

    // Проходим по пикселям на исходном изображении и создаем новое полутоновое изображение
    #pragma omp parallel for 
    for (int y = 0; y < height; y += 1) 
    {
        for (int x = 0; x < width; x += 1) 
        {
            // Получаем значения цветовых компонентов пикселей
            cv::Vec3b pixel = input_image.at<cv::Vec3b>(y, x);
            int red = pixel[2];   // Красная компонента
            int green = pixel[1]; // Зеленая компонента
            int blue = pixel[0];  // Синяя компонента

            // Вычисляем интенсивность
            int intensity = (red + green + blue) / 3;

            // Заполняем 2x2 область на новом изображении
            for (int i = 0; i < 2; i++) 
            {   
                for (int j = 0; j < 2; j++) 
                {
                    output_image.at<uchar>(y * 2 + j, x * 2 + i) = static_cast<uchar>(intensity);
                }
            }
        }
    }

    // Сохраняем результат в файл
    cv::imwrite("output_image2.jpg", output_image);

    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start1);

    cout << " OpenMP time: " << duration1.count() << " milliseconds" << endl;
    return 0;
}
