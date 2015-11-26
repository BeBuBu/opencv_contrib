#ifndef DSST_UTIL_HPP
#define DSST_UTIL_HPP

#include "precomp.hpp"
namespace cv{

    const float cn_data[] = {
  #include "ColorNames.txt"
    };

/*
*  internal representation of the tracked region
*/
struct TrackedRegion{

public:
    TrackedRegion(){ }
    TrackedRegion(const cv::Point2i init_center, const cv::Size init_size) : center(init_center), size(init_size){ }
    TrackedRegion(const cv::Rect box) : center(box.x + round((float)box.size().width /2.0),
     box.y + round((float)box.size().height/2.0)),
    size(box.size()){ }

    cv::Rect Rect() const{
        return cv::Rect(center.x-floor((float)size.width /2.0),
            center.y-floor((float)size.height/2.0),
            size.width,size.height);
    }

    TrackedRegion resize(const float factor) const{
        TrackedRegion newRegion;
        newRegion.center = center;
        newRegion.size = cv::Size(round(size.width *factor),
          round(size.height*factor));
        return newRegion;
    }

    cv::Point2i center;
    cv::Size size;
};


cv::Mat blockAverageImage(const cv::Mat& inputImage, const int cell_size);

void getFhog(const cv::Mat& inputImage, std::vector<cv::Mat>& featureVector, const int cell_size);

std::vector<cv::Mat> getColornames(cv::Mat image);

inline float getColorname(int cname,cv::Vec3b pixelValue);

void divSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB);

cv::Mat channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb);

std::vector<cv::Mat> extractFeatures(cv::Mat& image_patch, const std::vector<TrackerDSST::Feature> featureVec);

void emplaceFeature(cv::Mat& imagePatch, cv::Mat& featureMatrix, const int col, const float featureWeight,
    const std::vector<TrackerDSST::Feature> featureVec);

cv::Mat extractTrackedRegion(const cv::Mat image, const TrackedRegion region);

int totalFeatureDim(const std::vector<TrackerDSST::Feature> featureVec);

cv::Mat padSpectrum(const cv::Mat& spectrum, const cv::Size targetSize);

}
#endif