#include "fft.h"
#include <cmath>
#include <vector>
#include <complex>
#include <stdexcept>

using Complex = std::complex<double>;
using Real = double;

namespace {
    const Real PI = std::acos(-1.0);                
    const Complex J(0.0, 1.0);                      

    // для radix-3 
    const Real R3_SIN_2PI_3 = std::sin(2.0 * PI / 3.0); 
    const Real R3_C1 = -R3_SIN_2PI_3;                   

    // для radix-5 
    const Real R5_COS_4PI_5 = std::cos(4.0 * PI / 5.0); 
    const Real R5_SIN_4PI_5 = std::sin(4.0 * PI / 5.0); 
    const Real R5_SIN_2PI_5 = std::sin(2.0 * PI / 5.0); 
    const Real R5_K0 = R5_COS_4PI_5;
    const Real R5_K10 = -R5_SIN_4PI_5;
    const Real R5_T_CONST = R5_SIN_2PI_5;
    const Real R5_K11 = R5_T_CONST - R5_K10;
    const Real R5_K12 = -R5_T_CONST - R5_K10;

    inline bool is_power_of_two(size_t n) {
        return (n > 0) && ((n & (n - 1)) == 0);
    }


    size_t log2_power_of_two(size_t n) {
        if (n == 0) { 
             throw std::runtime_error("log2_power_of_two: n не может быть 0");
        }
        if (!is_power_of_two(n)) { 
             throw std::runtime_error("log2_power_of_two: n должно быть степенью двойки.");
        }
        if (n == 1) return 0;
        size_t count = 0;
        while (n > 1) {
            n >>= 1;  
            count++;
        }
        return count;
    }

    size_t reverse_num_bits(size_t n, size_t num_bits) {
        if (num_bits == 0) return 0; 
        size_t reversed_n = 0;
        for (size_t bit = 0; bit < num_bits; ++bit) {
            if ((n >> bit) & 1) { 
                reversed_n |= (1 << (num_bits - 1 - bit)); 
            }
        }
        return reversed_n;
    }

    void iota_manual(std::vector<size_t>& vec, size_t start_val) {
        for (size_t i = 0; i < vec.size(); ++i) 
            vec[i] = start_val + i;
    }

} 

void FFT::base_transform_radix2(Complex* y){
    Complex t = y[1];
    y[1] = y[0] - t;
    y[0] += t;
}

void FFT::base_transform_radix3(Complex* y) {
    Complex a = y[1] + y[2];
    Complex t1 = y[0] - a / 2.0;
    Complex t2 = (y[1] - y[2]) * R3_C1; 
    Complex b = ::J * t2;
    y[0] += a;
    y[1] = t1 + b;
    y[2] = t1 - b;
}

void FFT::base_transform_radix5(Complex* y) {
    Complex a1 = y[1] + y[4];
    Complex a2 = y[2] + y[3];
    Complex a3 = y[2] - y[3]; 
    Complex a4 = y[1] - y[4]; 
    
    Complex c5 = R5_K0 * (a1 - a2);
    Complex c6 = R5_K10 * (a3 + a4); 
    Complex d1 = y[0] - a1 / 2.0 - c5;
    Complex d2 = y[0] - a2 / 2.0 + c5;
    
    Complex t1 = ::J * (R5_K11 * a3 + c6); 
    Complex t2 = ::J * (R5_K12 * a4 + c6); 
    y[0] += a1 + a2;
    y[1] = d1 + t2;
    y[2] = d2 + t1;
    y[3] = d2 - t1; 
    y[4] = d1 - t2; 
}

FFT::FFT(size_t length) : N_(length) {
    if (N_ == 0) {
        radix_ = 0;
        num_stages_ = 0;
        base_transform_func_ = nullptr;
        return;
    }
    if (N_ == 1) {
        radix_ = 1; 
        num_stages_ = 0; 
        base_transform_func_ = nullptr;
        precompute_permutation_indices(); 
        return;
    }

    size_t M; 
    if (N_ % 5 == 0 && is_power_of_two(N_ / 5)) {
        radix_ = 5;
        M = N_ / 5;
        base_transform_func_ = FFT::base_transform_radix5;
    } else if (N_ % 3 == 0 && is_power_of_two(N_ / 3)) {
        radix_ = 3;
        M = N_ / 3;
        base_transform_func_ = FFT::base_transform_radix3;
    } else if (is_power_of_two(N_)) { 
        radix_ = 2;
        M = N_ / 2; 
        base_transform_func_ = FFT::base_transform_radix2;
    }
  
    num_stages_ = log2_power_of_two(M);

    precompute_twiddle_factors();
    precompute_permutation_indices();
}
void FFT::precompute_twiddle_factors() {
    if (N_ < 2) return; 

    twiddle_factors_.resize(N_ / 2);
    for (size_t k = 0; k < N_ / 2; ++k) {
        Real angle = -2.0 * PI * static_cast<Real>(k) / static_cast<Real>(N_);
        twiddle_factors_[k] = std::polar(1.0, angle); 
    }
}

void FFT::precompute_permutation_indices() {

    permutation_indices_.resize(N_);

    if (N_ == 0) return; 
    if (N_ == 1) {
        permutation_indices_[0] = 0;
        return;
    }
    size_t M_val = N_ / radix_; 
    size_t S_val = num_stages_; 

    for (size_t k = 0; k < N_; ++k) {
        size_t n1 = k / M_val;
        size_t n0 = k % M_val;
        permutation_indices_[k] = reverse_num_bits(n0, S_val) * radix_ + n1;
    }
}

std::vector<Complex> FFT::fft(const std::vector<Complex>& input) {
    if (input.size() != N_) { throw std::invalid_argument("fft: Размер входных данных не совпадает с N"); }
    if (N_ == 0) return {};
    if (N_ == 1) return input; 
    std::vector<Complex> data = input;
    core_fft(data, false); 
    return data;
}

std::vector<Complex> FFT::ifft(const std::vector<Complex>& input) {
    if (input.size() != N_) { throw std::invalid_argument("ifft: Размер входных данных не совпадает с N"); }
    if (N_ == 0) return {};
    if (N_ == 1) return input;
    std::vector<Complex> conj_input(N_);
    for (size_t i = 0; i < N_; ++i) {
        conj_input[i] = std::conj(input[i]);
    }
    core_fft(conj_input, true); 
    Real scale = 1.0 / static_cast<Real>(N_);
    std::vector<Complex> result(N_);
    for (size_t i = 0; i < N_; ++i) {
        result[i] = std::conj(conj_input[i]) * scale;
    }
    return result;
}


void FFT::core_fft(std::vector<Complex>& data, bool) {

    if (N_ > 1) { 
        std::vector<Complex> temp_data = data;
        for (size_t i = 0; i < N_; ++i) {
            if (permutation_indices_[i] >= N_) { 
                throw std::out_of_range("core_fft: Индекс перестановки вне диапазона.");
            }
            data[permutation_indices_[i]] = temp_data[i];
        }
    }



    if (radix_ > 1 && base_transform_func_ != nullptr) {
        size_t M_val = N_ / radix_; 
        for (size_t m_idx = 0; m_idx < M_val; ++m_idx) {
            base_transform_func_(&data[m_idx * radix_]);
        }
    }

    
    size_t M_len = N_ / radix_; 
    size_t current_processing_block_size_nr = radix_; 
    size_t num_butterfly_sets_nl = M_len / 2; 
    for (size_t s = 0; s < num_stages_; ++s) { 
        for (size_t l_set_idx = 0; l_set_idx < num_butterfly_sets_nl; ++l_set_idx) {
            for (size_t r_butterfly_leg_idx = 0; r_butterfly_leg_idx < current_processing_block_size_nr; ++r_butterfly_leg_idx) {
                size_t p_offset = 2 * l_set_idx * current_processing_block_size_nr;
                size_t idx_p = p_offset + r_butterfly_leg_idx;
                size_t idx_q = idx_p + current_processing_block_size_nr;
                size_t twiddle_lut_idx = r_butterfly_leg_idx * num_butterfly_sets_nl;
                Complex Wk = twiddle_factors_[twiddle_lut_idx];
                Complex t = data[idx_q] * Wk;
                data[idx_q] = data[idx_p] - t;
                data[idx_p] += t;
            }
        }
        current_processing_block_size_nr *= 2;
        num_butterfly_sets_nl /= 2;
    }
}