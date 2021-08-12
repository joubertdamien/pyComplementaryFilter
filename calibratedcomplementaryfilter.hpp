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
#include "complementaryfilter.hpp"
#ifndef CALIBRATEDFILTER_HPP_
#define CALIBRATEDFILTER_HPP_
class CalibratedCFilter{
    private:
        std::vector<float_t> last_log_image_;
        std::vector<float_t> cur_log_image_;
        std::vector<float_t> ths_pos_;
        std::vector<float_t> ths_neg_;
        std::vector<float_t> alp_ar_;
        std::vector<uint64_t> tsurface_;
        uint16_t x_;
        uint16_t y_;
        float_t alp_;
        float_t lam_;
        uint16_t L1_;
        uint16_t L2_;
        uint16_t maxL_;
        logLUT lut_;
    public:
        CalibratedCFilter(uint16_t& x, uint16_t& y, float_t& th_pos, float_t& th_neg, float_t& alp, float_t& lam, 
            uint16_t& L1, uint16_t& L2, uint16_t& maxL, std::vector<float_t>& ths_pos, std::vector<float_t>& ths_neg){
            tsurface_.resize(x * y, 0);
            cur_log_image_.resize(x * y, 0);
            last_log_image_.resize(x * y, 0);
            alp_ar_.resize(x * y, alp);
            ths_pos_ = ths_pos;
            ths_neg_ = ths_neg;
            x_ = x;y_ = y;
            alp_ = alp;
            lam_ = lam;
            L1_ = L1;
            L2_ = L2;
            maxL_ = maxL;
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
                cur_log_image_.at(p) += (it->is_increase) ? ths_pos_.at(p) : ths_neg_.at(p);
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
                    alp_ar_.at(p) = alp_ * ( lam_ + (1-lam_) * it->intensity / L1_);
                else{
                    if(it->intensity > L2_)
                        alp_ar_.at(p) = alp_ * ( lam_ + (1-lam_) * (it->intensity- maxL_) / (L2_ - maxL_));
                    else
                        alp_ar_.at(p) = alp_;
                }
            }
        };
};
#endif 