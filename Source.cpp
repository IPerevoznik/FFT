#include "fft.h"
#include <iostream>
#include <vector>
#include <complex>
#include <random>     
#include <cmath>      
#include <limits>    
#include <iomanip>   
#include <vector>
#include <string> 
#include <fstream>

//#define LOG_ON



using Complex = std::complex<double>;
using Real = double;

std::vector<Complex> generate_random_complex_vector(size_t size, Real min_val = -1.0, Real max_val = 1.0) {
    std::vector<Complex> vec(size);
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<Real> distrib(min_val, max_val);

    for (size_t i = 0; i < size; ++i) {
        Real real_part = distrib(gen);
        Real imag_part = distrib(gen);
        vec[i] = Complex(real_part, imag_part);
    }
    return vec;
}

Real calculate_max_error(const std::vector<Complex>& v1, const std::vector<Complex>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("calculate_max_error: Размеры векторов должны совпадать.");
    }
    if (v1.empty()) {
        return 0.0;
    }

    Real max_err = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        Real current_err = std::abs(v1[i] - v2[i]);
        if (current_err > max_err) {
            max_err = current_err;
        }
    }
    return max_err;
}


Real calculate_rmse(const std::vector<Complex>& v1, const std::vector<Complex>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("calculate_rmse: Размеры векторов должны совпадать.");
    }
    size_t n = v1.size();
    if (n == 0) {
        return 0.0;
    }

    Real sum_sq_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        Real err = std::norm(v1[i] - v2[i]); 
        sum_sq_err += err;
    }
    return std::sqrt(sum_sq_err / n);
}


int main() {
    #ifdef LOG_ON
    std::ofstream logfile("fft_test_output.txt");
    if (!logfile.is_open()) {
        std::cerr << "Не удалось открыть файл для записи" << std::endl;
        return 1;
    }
   #endif


    std::cout << "Тест FFT класса" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    std::vector<size_t> test_lengths = {
        
         15, 25, 10, 2, 3, 5, 16, 125, 100, 90, 7, 54

 
    };

    std::cout << std::fixed << std::setprecision(std::numeric_limits<Real>::max_digits10);

#ifdef LOG_ON
logfile << std::fixed << std::setprecision(std::numeric_limits<Real>::max_digits10);
#endif

for (size_t length : test_lengths) {
    std::cout << "\nТест для N = " << length << ":" << std::endl;
    #ifdef LOG_ON
    logfile << "\nТест для N = " << length << ":" << std::endl;
   #endif

    try {
        FFT processor(length);
        std::vector<Complex> original_data = generate_random_complex_vector(length);

        if (length <= 15) {
            std::cout << "  Входные данные:" << std::endl;
            #ifdef LOG_ON
            logfile << "  Входные данные:" << std::endl;
          #endif
            for(const auto& val : original_data) {
                std::cout << "    " << val << std::endl;
                #ifdef LOG_ON
                logfile << "    " << val << std::endl;
               #endif
            }
        }

        std::vector<Complex> transformed_data = processor.fft(original_data);

        if (length <= 15) {
            std::cout << "  Результат FFT:" << std::endl;
            #ifdef LOG_ON
            logfile << "  Результат FFT:" << std::endl;
          #endif
            for(const auto& val : transformed_data) {
                std::cout << "    " << val << std::endl;
                #ifdef LOG_ON
                logfile << "    " << val << std::endl;
               #endif
            }
        }

        std::vector<Complex> reconstructed_data = processor.ifft(transformed_data);
        std::cout << " IFFT :" << std::endl;
        #ifdef LOG_ON
        logfile << " IFFT :" << std::endl;
       #endif

        if (length <= 15) {
            std::cout << "  Восстановленные данные:" << std::endl;
            #ifdef LOG_ON
            logfile << "  Восстановленные данные:" << std::endl;
          #endif
            for(const auto& val : reconstructed_data) {
                std::cout << "    " << val << std::endl;
                #ifdef LOG_ON
                logfile << "    " << val << std::endl;
                #endif
            }
        }

        Real max_error = calculate_max_error(original_data, reconstructed_data);
        Real rmse = calculate_rmse(original_data, reconstructed_data);

        std::cout << "  Сравнение оригинала и результата после FFT->IFFT:" << std::endl;
        #ifdef LOG_ON
        logfile << "  Сравнение оригинала и результата после FFT->IFFT:" << std::endl;
       #endif

        std::cout << std::scientific;
        #ifdef LOG_ON
        logfile << std::scientific;
        #endif

        std::cout << "    Максимальная абсолютная ошибка: " << max_error << std::endl;
        #ifdef LOG_ON
        logfile << "    Максимальная абсолютная ошибка: " << max_error << std::endl;
       #endif

        std::cout << "    Среднеквадратичная ошибка : " << rmse << std::endl;
        #ifdef LOG_ON
        logfile << "    Среднеквадратичная ошибка : " << rmse << std::endl;
       #endif

        std::cout << std::fixed;
        #ifdef LOG_ON
        logfile << std::fixed;
       #endif

    } catch (const std::invalid_argument& e) {
        std::cerr << "  не та N" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  где то ошибся: " << e.what() << std::endl;
    }
}

    return 0;
}