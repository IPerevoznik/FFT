#include "fft.h"
#include <cmath>
#include <vector>
#include <complex>
#include <stdexcept>
#include <string>
#include <algorithm> 
#include <iostream> 
#include <iomanip>  

namespace { 
    const double PI = std::acos(-1.0); 
}

void FFT::base_transform_radix2(Complex* y, bool inverse) { 
    Complex x0 = y[0]; 
    Complex x1 = y[1]; 
    y[0] = x0 + x1; 
    y[1] = x0 - x1;
}

void FFT::base_transform_radix3(Complex* y, bool inverse) { 
    Real angle_sign = inverse ? 1.0 : -1.0;
    Complex W3_1 = std::polar(1.0, angle_sign * 2.0 * PI / 3.0);
    Complex W3_2 = std::polar(1.0, angle_sign * 4.0 * PI / 3.0);
    Complex x0 = y[0]; 
    Complex x1 = y[1];
    Complex x2 = y[2];
    Complex Y0 = x0 + x1 + x2; 
    Complex Y1 = x0 + x1*W3_1 + x2*W3_2; 
    Complex Y2 = x0 + x1*W3_2 + x2*W3_1;
    y[0] = Y0; 
    y[1] = Y1;
    y[2] = Y2;
}

void FFT::base_transform_radix5(Complex* y, bool inverse) { 
    Real angle_sign = inverse ? 1.0 : -1.0;
    Complex W5_1 = std::polar(1.0, angle_sign * 2.0 * PI / 5.0);
    Complex W5_2 = std::polar(1.0, angle_sign * 4.0 * PI / 5.0);
    Complex W5_3 = std::polar(1.0, angle_sign * 6.0 * PI / 5.0);
    Complex W5_4 = std::polar(1.0, angle_sign * 8.0 * PI / 5.0);
    Complex x0 = y[0], x1 = y[1], x2 = y[2], x3 = y[3], x4 = y[4];
    Complex Y0 = x0 + x1 + x2 + x3 + x4;
    Complex Y1 = x0 + x1*W5_1 + x2*W5_2 + x3*W5_3 + x4*W5_4;
    Complex Y2 = x0 + x1*W5_2 + x2*W5_4 + x3*W5_1 + x4*W5_3;
    Complex Y3 = x0 + x1*W5_3 + x2*W5_1 + x3*W5_4 + x4*W5_2;
    Complex Y4 = x0 + x1*W5_4 + x2*W5_3 + x3*W5_2 + x4*W5_1;
    y[0] = Y0; 
    y[1] = Y1; 
    y[2] = Y2; 
    y[3] = Y3; 
    y[4] = Y4;
}


FFT::FFT(size_t length) : N_(length) { 
    if (N_ == 0) 
    { 
        return; 
    }
    prime_factors_sequence_.clear();
    size_t temp_N = N_;
    while (temp_N > 1 && temp_N % 2 == 0){ 
        prime_factors_sequence_.push_back(2); 
        temp_N /= 2; 
    }
    while (temp_N > 1 && temp_N % 3 == 0) { 
        prime_factors_sequence_.push_back(3);
         temp_N /= 3;
    }
    while (temp_N > 1 && temp_N % 5 == 0) { 
        prime_factors_sequence_.push_back(5);
        temp_N /= 5; 
    }
    if (temp_N > 1) { 
        throw std::invalid_argument("FFT: содержит множители, отличные от 2, 3, 5") ; 
    }
    if (N_ > 1 && prime_factors_sequence_.empty()) {
         throw std::invalid_argument("FFT: Не удалось разложить N "); 
    }
    precompute_twiddle_factors();
    precompute_permutation_indices();
}

void FFT::precompute_twiddle_factors() { 
    if (N_ < 2) { 
        twiddle_factors_.clear(); 
        return; 
    }

    twiddle_factors_.resize(N_);
    for (size_t k = 0; k < N_; ++k) {
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

    for (size_t k = 0; k < N_; ++k) {
        size_t original_val = k;
        size_t permuted_val = 0;
        for (auto it = prime_factors_sequence_.rbegin(); it != prime_factors_sequence_.rend(); ++it) {
            size_t factor = *it;
            permuted_val = permuted_val * factor + (original_val % factor);
            original_val /= factor;
        }
        permutation_indices_[k] = permuted_val;
    }
}


std::vector<Complex> FFT::fft(const std::vector<Complex>& input) { 
    if (input.size() != N_) { throw std::invalid_argument("fft: Размер не совпадает."); }
    if (N_ == 0) return {};
    std::vector<Complex> data = input; 
    core_fft(data, false); 
    return data;
}

std::vector<Complex> FFT::ifft(const std::vector<Complex>& input) { 
    if (input.size() != N_) { throw std::invalid_argument("ifft: Размер не совпадает."); }
    if (N_ == 0) return {};
    std::vector<Complex> data = input; 
    core_fft(data, true); 
    if (N_ > 0) { 
        Real scale = 1.0 / static_cast<Real>(N_); 
        for (size_t i = 0; i < N_; ++i) 
        { 
            data[i] *= scale;
        } 
        }
    return data;
}


void FFT::core_fft(std::vector<Complex>& data, bool inverse) { 
    if (N_ <= 1) return;

    std::vector<Complex> temp_perm_data = data;
    for (size_t i = 0; i < N_; ++i) { 
        data[permutation_indices_[i]] = temp_perm_data[i]; 
    }
    
    size_t current_processed_block_size = 1; 


    for (size_t stage_radix : prime_factors_sequence_) {
        void(*selected_base_transform_func)(Complex*, bool); 
        if (stage_radix == 2) selected_base_transform_func = FFT::base_transform_radix2;
        else if (stage_radix == 3) selected_base_transform_func = FFT::base_transform_radix3;
        else if (stage_radix == 5) selected_base_transform_func = FFT::base_transform_radix5;
        else { ; 
        }
        
        size_t num_dft_groups = N_ / (stage_radix * current_processed_block_size); 
        
        for (size_t group_idx = 0; group_idx < num_dft_groups; ++group_idx) {
            for (size_t element_in_prev_block_idx = 0; element_in_prev_block_idx < current_processed_block_size; ++element_in_prev_block_idx) {
                std::vector<Complex> y_buffer(stage_radix); 
                
                for (size_t leg_idx = 0; leg_idx < stage_radix; ++leg_idx) {
                    size_t data_idx = group_idx * stage_radix * current_processed_block_size + 
                                      leg_idx * current_processed_block_size +                 
                                      element_in_prev_block_idx;                             
                    y_buffer[leg_idx] = data[data_idx];
                    
                    if (leg_idx > 0) { 
                        
                        size_t twiddle_idx = (leg_idx * element_in_prev_block_idx * num_dft_groups) % N_;
                        Complex W = twiddle_factors_[twiddle_idx]; 
                        if (inverse) {
                             W = std::conj(W); 
                            }
                        y_buffer[leg_idx] *= W;
                    }
                }
                
                selected_base_transform_func(y_buffer.data(), inverse); 
                
                for (size_t leg_idx = 0; leg_idx < stage_radix; ++leg_idx) {
                    size_t data_idx = group_idx * stage_radix * current_processed_block_size +
                                      leg_idx * current_processed_block_size +
                                      element_in_prev_block_idx;
                    data[data_idx] = y_buffer[leg_idx];
                }
            }
        }
        current_processed_block_size *= stage_radix;
    }
}


