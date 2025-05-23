#ifndef FFT_
#define FFT_

#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>

using Complex = std::complex<double>;
using Real = double;

class FFT {
public:
    /** 
     * @brief создает объект для преобразование значении определенной длины
     *  @param length длина преобразования N, которая может быть кратной 2,3,5
     */
    FFT (size_t length);


    /** @brief прямое быстрое преобразование (fft)
     *  @param input входной вектор, размер должен совпадать с length в конструкторе
     *  @return вектор с результатами FFT
     */
    std::vector<Complex> fft(const std::vector<Complex>&input);

    /** @brief обратное быстрое преобразование (ifft)
     * 
     */
    std::vector<Complex> ifft(const std::vector<Complex>& input);

    /**
     * @brief длина преобразования (N)
     * @return возвращает длину преобразования
     */
    size_t get_length() const { return N_; }

private:
    size_t N_; // длина преобразования

    std::vector<size_t> prime_factors_sequence_; 

    std::vector<Complex> twiddle_factors_; // предвычисленные W_N^k 
    std::vector<size_t> permutation_indices_; // предвычисленные индексы для смешанной перестановки

    // вычисление во время инициализации
    void precompute_twiddle_factors();
    void precompute_permutation_indices();

    static void base_transform_radix2(Complex* buffer, bool inverse);
    static void base_transform_radix3(Complex* buffer, bool inverse);
    static void base_transform_radix5(Complex* buffer, bool inverse);
    

    // основная функция преобразования
    void core_fft(std::vector<Complex>& data, bool inverse);

};



#endif

   