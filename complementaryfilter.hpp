#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <cstring>
#include <numpy/arrayobject.h>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <limits>
#include <memory>
#include "sepia/source/sepia.hpp"
#ifndef FILTER_HPP_
#define FILTER_HPP_
using Event = sepia::event<sepia::type::dvs>;
struct __attribute__((__packed__)) IntensityEvent{
    uint64_t t;
    uint16_t x;
    uint16_t y;
    float_t intensity;
};
class logLUT{
    private:
        uint16_t maxVal_;
        std::vector<float> lut_;
    public:
        logLUT(){};
        logLUT(uint16_t maxVal){
            maxVal_ = maxVal;
            lut_.resize(maxVal);
            for(uint16_t i = 0; i < maxVal_; i++){
                lut_.at(i) = std::log(i);
            }
        }
        float_t getVal(uint16_t t){
            return (t == 0 || t >= maxVal_) ? 0 : lut_.at(t);
        }
};
class CFilter{
    private:
        std::vector<float_t> last_log_image_;
        std::vector<float_t> cur_log_image_;
        std::vector<float_t> alp_ar_;
        std::vector<uint64_t> tsurface_;
        uint16_t x_;
        uint16_t y_;
        float_t th_pos_;
        float_t th_neg_;
        float_t alp_;
        float_t lam_;
        float_t L1_;
        float_t L2_;
        float_t maxL_;
        logLUT lut_;
    public:
        CFilter(uint16_t& x, uint16_t& y, float_t& th_pos, float_t& th_neg, float_t& alp, float_t& lam, uint16_t& L1, uint16_t& L2, uint16_t& maxL){
            tsurface_.resize(x * y, 0);
            cur_log_image_.resize(x * y, 0);
            last_log_image_.resize(x * y, 0);
            alp_ar_.resize(x * y, alp);
            x_ = x;y_ = y;
            th_pos_ = th_pos;th_neg_ = -th_neg;
            alp_ = alp;
            lam_ = lam;
            L1_ = (L1 > 0) ? std::log(L1) : 0;
            L2_ = (L2 > 0) ? std::log(L2) : 0;
            maxL_ = (maxL > 0) ? std::log(maxL) : 0;
            lut_ = logLUT(maxL);
        };
        uint32_t getPos(const uint16_t& x, const uint16_t& y){return y * x_ + x;}
        void processEvent(std::vector<Event>& in, std::vector<IntensityEvent>& out){
            out.reserve(in.size());
            uint32_t p;
            float_t dec;
            for(auto it = in.begin(); it < in.end(); it++){
                p = getPos(it->x, it->y);
                dec = std::exp(-static_cast<float_t>(it->t - tsurface_.at(p)) * alp_ar_.at(p));
                cur_log_image_.at(p) = dec * cur_log_image_.at(p) + (1-dec) * last_log_image_.at(p);
                cur_log_image_.at(p) += (it->is_increase) ? th_pos_ : th_neg_;
                out.push_back(IntensityEvent{it->t, it->x, it->y, cur_log_image_.at(p)});
                tsurface_.at(p) = it->t;
            }
        };
        void processIntensityEvent(std::vector<IntensityEvent>& in, std::vector<IntensityEvent>& out){
            uint32_t p;
            float_t dec;
            for(auto it = in.begin(); it < in.end(); it++){
                p = getPos(it->x, it->y);
                dec = std::exp(-static_cast<float_t>(it->t - tsurface_.at(p)) * alp_ar_.at(p));
                cur_log_image_.at(p) = dec * cur_log_image_.at(p) + (1-dec) * last_log_image_.at(p);
                out.push_back(IntensityEvent{it->t, it->x, it->y, cur_log_image_.at(p)});
                tsurface_.at(p) = it->t;
                last_log_image_.at(p) = (it->intensity > 0) ? lut_.getVal(it->intensity) : 0;
                if(it->intensity < L1_)
                    alp_ar_.at(p) = alp_ * ( lam_ + (1-lam_) * last_log_image_.at(p) / L1_);
                else{
                    if(it->intensity > L2_)
                        alp_ar_.at(p) = alp_ * ( lam_ + (1-lam_) * (last_log_image_.at(p) - maxL_) / (L2_ - maxL_));
                    else
                        alp_ar_.at(p) = alp_;
                }
            }
        };
};
#endif 