#include "precomp.hpp"

#include "dsst_util.hpp"

namespace cv{


//Adaptive Correlation Filter
    class ACF{
    public:
        ACF(const TrackerDSST::Params& p) : params(p)  { }

        virtual void initialize(const cv::Mat image, const TrackedRegion) = 0;
        virtual ~ACF() {}
        virtual void learn(const cv::Mat image, const TrackedRegion target) = 0;
        virtual void track(const cv::Mat image, TrackedRegion& target) = 0;

    protected:
      cv::Mat yf;
      cv::Size filterSize;

      TrackerDSST::Params params;
  //the acual filter parameters
      cv::Mat den;
  //std::vector<cv::Mat> num;

      virtual void generateConstants(const cv::Size filterSize) = 0;
  };

  class TranslationFilter : public ACF{
  public:
    TranslationFilter(const TrackerDSST::Params& p);
    ~TranslationFilter() {}
    void learn(const cv::Mat image,const TrackedRegion target);
    void track(const cv::Mat image, TrackedRegion& target);
    void initialize(const cv::Mat image, const TrackedRegion target);

private:
    void generateConstants(const cv::Size filterSize);
    cv::Mat window;
    std::vector<cv::Mat> num;
    float filterDownscaleFactor;

    cv::Mat trackPatch, trainPatch;
};

class ScaleFilter : public ACF{
public:
    ScaleFilter(const TrackerDSST::Params& p, const int nScales,const float scaleStep);
    ~ScaleFilter() {}
    void learn(const cv::Mat image, const TrackedRegion target);
    void track(const cv::Mat image, TrackedRegion& target);
    void initialize(const cv::Mat image, const TrackedRegion target);
private:
    void generateConstants(const cv::Size filterSize);
    int nScales;
    float scaleStep;
    float maxScale,minScale;
    std::vector<float> scaleFactors;
    std::vector<float> window;
    cv::Mat num;

    cv::Size initialSize;
    float currentScaleFactor;
    float scaleModelFactor;

  /*
   * -----------------------------------------
   *  Allocate memory statically for parts of algorithm
   *  ---------------------------------------
   */

  //vector to hold the new scale features in the learning step
   cv::Mat scaleFeatureData_;
   cv::Mat scaleSample_;
};

class VisualTracker{
public:
    VisualTracker(const TrackerDSST::Params& p);
    void addFilter(ACF* f);
    void track(const cv::Mat image);
    void learn(const cv::Mat image);
    void reset(const cv::Mat image, const cv::Rect initial_bb);
    cv::Rect getBoundingBox(void) const;
    void startTracker(const cv::Mat image, const cv::Rect initial_bb);
private:
    TrackedRegion target;
    std::vector<ACF*> filters;
};


class DSST : public TrackerDSST{
public:
    DSST(const TrackerDSST::Params&p) : tracker(p) {

    }
    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;

    bool initImpl( const Mat& /*image*/, const Rect2d& boundingBox );
    bool updateImpl( const Mat& image, Rect2d& boundingBox );

private:
    VisualTracker tracker;
    TrackerDSST::Params params;
};

VisualTracker::VisualTracker(const TrackerDSST::Params& p){


    filters.push_back(new TranslationFilter(p));


    //fixed parameters for the scale filter as they have little influence
    //on results
    TrackerDSST::Params scale_settings;
    scale_settings.sigma = 1.0/16.0;
    scale_settings.learningRate = 0.025;
    scale_settings.lambda = 0.01;
    scale_settings.features.push_back(TrackerDSST::Gray);
    scale_settings.padding = 0.0;
    scale_settings.minPatchSize = cv::Size(16,16);
    scale_settings.maxPatchSize = cv::Size(32,32);
    scale_settings.maxFilterArea = 16*32;
    scale_settings.minFilterArea = 10*10;
    filters.push_back(new ScaleFilter(scale_settings,33,(double)1.02));
}


void DSST::read(const FileNode& /*fn*/){

}

void DSST::write(FileStorage& /*fs*/) const {

}
/*
 * Constructor
 */
 Ptr<TrackerDSST> TrackerDSST::createTracker(const DSST::Params &parameters){
  return Ptr<DSST>(new DSST(parameters));
}

bool DSST::initImpl(const Mat& image, const Rect2d& boundingBox){
    tracker.startTracker(image, boundingBox);
    return true;
}

bool DSST::updateImpl(const Mat& image, Rect2d& boundingBox){
    tracker.track(image);
    tracker.learn(image);
    boundingBox = tracker.getBoundingBox();
    return true;
}

/*
 *  The filter is started by discarding all prevous measurments and using only
 *  the current data to update the filter state
 */
 void TranslationFilter::initialize(const cv::Mat image, const TrackedRegion target){

    TrackedRegion paddedRegion = target.resize(1.0 + params.padding);

    if(paddedRegion.size.area() > params.maxFilterArea){
        filterDownscaleFactor = sqrt((float)paddedRegion.size.area() / (float)params.maxFilterArea);
    }else{
        filterDownscaleFactor = 1.0;
    }

    filterSize = cv::Size(paddedRegion.size.width/filterDownscaleFactor,
      paddedRegion.size.height/filterDownscaleFactor);

    //compute the minimum and maximum sizes for the tracker
    den = cv::Mat(filterSize,CV_32FC1);
    generateConstants(filterSize);

    //set the initial filter parameters
    cv::Mat patch = extractTrackedRegion(image, paddedRegion);

    //resize the tracked patch to have the same size as the filter before extracting features
    cv::resize(patch,patch,filterSize);

    std::vector<cv::Mat> featurePatch = extractFeatures(patch,params.features);

    std::vector<cv::Mat> xf = std::vector<cv::Mat>(featurePatch.size());

    for(unsigned int i=0; i < featurePatch.size(); ++i){
        cv::dft(featurePatch[i].mul(window),xf[i],cv::DFT_COMPLEX_OUTPUT);
    }
    den = channelMultiply(xf,xf,0,true);

    //label function multiplied by x
    num = std::vector<cv::Mat>(featurePatch.size());

    for(unsigned int i=0; i < xf.size(); ++i){
        mulSpectrums(yf,xf[i],num[i],0,true);
    }

}

void TranslationFilter::generateConstants(const cv::Size matrixSize){
    //create the labeling weights for use in training the classifier

    cv::Size trackerSize;
    if(params.features[0] == TrackerDSST::HOG  || params.features[0] == TrackerDSST::FHOG4I ){
        trackerSize = cv::Size(round(matrixSize.width/4.0)-2,
         round(matrixSize.height/4.0)-2);
    }else{
        trackerSize = matrixSize;
    }

    cv::Mat y(trackerSize,CV_32FC1);

    float r, c;
    int fszw2 = floor(trackerSize.width / 2.0);
    int fszh2 = floor(trackerSize.height / 2.0);

    double sigmasquared = -0.5 / (pow(params.sigma,2) * trackerSize.area()/pow(1.0 + params.padding,2));
    for (int i = 0; i < trackerSize.width; i++) {
        r = ((i + fszw2) % trackerSize.width) - fszw2;
        for (int j = 0; j < trackerSize.height; j++) {
            c = ((j + fszh2) % trackerSize.height) - fszh2;
            y.ptr<float>(j)[i] = (float)(sigmasquared * (pow(r,2) + pow(c,2)));
        }
    }

    cv::exp(y,y);
    dft(y, yf,cv::DFT_COMPLEX_OUTPUT);

    cv::createHanningWindow(window,trackerSize,CV_32FC1);

    cv::Point2i maxpt;
    cv::minMaxLoc(y,NULL,NULL,NULL,&maxpt);
}

void TranslationFilter::learn(const cv::Mat image,const TrackedRegion target){


    if(params.learningRate == 0){
        return;
    }

    TrackedRegion scaledRegion = target.resize(1.0 + params.padding);

    cv::Mat patch = extractTrackedRegion(image, scaledRegion);
    //imshow("Translation sample", patch);

    //resize the tracked patch to have the same size as the filter before extracting features
    cv::resize(patch,patch,filterSize);

    trainPatch = patch;

    std::vector<cv::Mat> featurePatch = extractFeatures(patch,params.features);
    std::vector<cv::Mat> xf = std::vector<cv::Mat>(featurePatch.size());
    std::vector<cv::Mat> new_num = std::vector<cv::Mat>(featurePatch.size());

    for(unsigned int i=0; i < featurePatch.size(); ++i){
        cv::dft(featurePatch[i].mul(window),xf[i],cv::DFT_COMPLEX_OUTPUT);
    }

    //this is the autocorrelation over all the channels
    cv::Mat new_den = channelMultiply(xf,xf,0,true);

    //label function multiplied by x generates the numerator
    for(unsigned int i=0; i < xf.size(); ++i){
        cv::mulSpectrums(yf,xf[i],new_num[i],0,true);
    }
    //update the denominator
    den = (1 - params.learningRate)*den + params.learningRate*new_den;

    //update the numerator
    for(unsigned int i=0; i < new_num.size(); ++i){
        num[i] = (1 - params.learningRate)*num[i] + params.learningRate*new_num[i];
    }
}

void TranslationFilter::track(const cv::Mat image, TrackedRegion& target){

    //add the padding to the bounding box
    TrackedRegion paddedRegion = target.resize(1.0 + params.padding);

    cv::Mat patch = extractTrackedRegion(image,paddedRegion);

    cv::resize(patch,patch,filterSize);

    std::vector<cv::Mat> featurePatch = extractFeatures(patch,params.features);
    std::vector<cv::Mat> xf = std::vector<cv::Mat>(featurePatch.size());

    for(unsigned int i=0; i < featurePatch.size(); ++i){
        cv::dft(featurePatch[i].mul(window),xf[i],cv::DFT_COMPLEX_OUTPUT);
    }

    cv::Mat kf = channelMultiply(xf,num,0,false);

    divSpectrums(kf,den+params.lambda,kf,0,false);

    double maxval;
    cv::Point2i maxpoint;
    cv::Mat response;
    cv::Point2i translation;
    if(params.interpolate_to_grid == true){
        cv::Mat padded_kf;
        padded_kf = padSpectrum(kf,paddedRegion.size);
        cv::dft(padded_kf, response, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    }else{
        cv::dft(kf, response, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    }

    cv::minMaxLoc(response,NULL,&maxval,NULL,&maxpoint);

    int fszw2 = floor(response.size().width / 2.0);
    int fszh2 = floor(response.size().height / 2.0);

    //convert from the (possibly) downsampled coordinates back to the image coordinate frame
    float scaleFactor = ((float)paddedRegion.size.width/(float)response.size().width);

    translation.x = round( scaleFactor*(((maxpoint.x + fszw2) % response.size().width) - fszw2));
    translation.y = round( scaleFactor*(((maxpoint.y + fszh2) % response.size().height) - fszh2));

    target.center = target.center + translation;
}

TranslationFilter::TranslationFilter(const TrackerDSST::Params& p) : ACF(p){

}

ScaleFilter::ScaleFilter(const TrackerDSST::Params& p, const int init_nScales, const float init_scaleStep) : ACF(p), nScales(init_nScales), scaleStep(init_scaleStep){

}

void ScaleFilter::generateConstants(const cv::Size init_filterSize){

    window = std::vector<float>(nScales);
    for(int n=0; n < nScales; ++n){
        window[n] = pow( sin(M_PI*n / (nScales - 1.0)) ,2);
    }
    window[nScales-1] = 0; //set to zero to avoid numerical strangeness

    cv::Mat y(1,nScales,CV_32FC1);
    //Matlab style indexing
    for(int s = 1; s <= nScales; ++s ){
        float ss = s - ceil(nScales/2.0);

        y.at<float>(0,s-1) = exp(-0.5 * pow(ss,2) / pow(params.sigma * nScales,2));
    }

    cv::dft(y, yf, cv::DFT_ROWS);

    //pre compute all the scale factors relative to the current scale we are interested in
    scaleFactors = std::vector<float>(nScales);
    for(int scale=0; scale < nScales; ++scale){
        float exponent = ceil(nScales/2.0) - (scale+1.0);
        scaleFactors[scale] = (float)pow(scaleStep,exponent);
    }
}

void ScaleFilter::initialize(const cv::Mat image, const TrackedRegion target){

    initialSize = cv::Size(target.size);
    currentScaleFactor = 1.0;
    scaleModelFactor = 1.0;

    minScale = pow(scaleStep,ceil(std::log(std::max(5.0/target.size.width,5.0/target.size.height)) /
      std::log(scaleStep)));

    maxScale = pow(scaleStep,
        floor(std::log(std::min(image.size().width/target.size.width,
            image.size().height/target.size.height)) /
        std::log(scaleStep)));

    if(pow(scaleModelFactor,2) * target.size.area() > params.maxFilterArea){
        scaleModelFactor = sqrt((float)params.maxFilterArea / (float)target.size.area());
    }
    filterSize = cv::Size(floor((float)target.size.width *scaleModelFactor),
      floor((float)target.size.height*scaleModelFactor));


    if(params.features[0] == TrackerDSST::FHOG1 || params.features[0] == TrackerDSST::FHOG4){
        //scale filter feature size
        cv::Size sffs = cv::Size(round(filterSize.width/4.0)-2,
            round(filterSize.height/4.0)-2);
        const int nScaleFeatureDims = sffs.area() * 31;
        scaleFeatureData_ = cv::Mat(nScaleFeatureDims,nScales,CV_32FC1,cv::Scalar(100.0));
    }else{
        //allocate memory for the feature
        const int nFeatureDimensions = totalFeatureDim(params.features);
        scaleFeatureData_ = cv::Mat(filterSize.area()*nFeatureDimensions,nScales,CV_32FC1,cv::Scalar(100.0));
    }

    generateConstants(filterSize);

    //setup the initial model
    for(int scale = 0; scale < nScales; ++scale){
        float scaleFactor = scaleFactors[scale];
        TrackedRegion scaledRegion(target.center,initialSize);
        scaledRegion = scaledRegion.resize(scaleFactor*currentScaleFactor);
        cv::Mat subimage = extractTrackedRegion(image,scaledRegion);

        //resize it to be the same size as the filter
        cv::resize(subimage, scaleSample_, filterSize);

        float windowFactor = window[scale];
        emplaceFeature(scaleSample_,scaleFeatureData_,scale,windowFactor,params.features);
    }

    cv::Mat xf,den_split;
    cv::dft(scaleFeatureData_,xf,cv::DFT_ROWS);
    cv::mulSpectrums(xf,xf,den_split,cv::DFT_ROWS,true);
    cv::reduce(den_split,den,0,CV_REDUCE_SUM);

    num = cv::Mat(scaleFeatureData_.size(),CV_32FC1);
    for(int i=0; i < xf.rows; i++){
        mulSpectrums(yf,xf.row(i),num.row(i),cv::DFT_ROWS,true);
    }
}

void ScaleFilter::track(const cv::Mat image, TrackedRegion& target){

    for(int scale=0; scale < nScales; ++scale){
        float scaleFactor = scaleFactors[scale];

        TrackedRegion scaledRegion = TrackedRegion(target.center,initialSize);
        scaledRegion = scaledRegion.resize(scaleFactor*currentScaleFactor);

        cv::Mat subimage = extractTrackedRegion(image,scaledRegion);

        cv::resize(subimage, scaleSample_,filterSize);
        float windowFactor = window[scale];
        emplaceFeature(scaleSample_,scaleFeatureData_,scale,windowFactor,params.features);
    }

    cv::Mat xf,kf,kf_full,response,responsef;
    cv::dft(scaleFeatureData_,xf,cv::DFT_ROWS);
    cv::mulSpectrums(xf,num,kf_full,cv::DFT_ROWS,false);
    cv::reduce(kf_full,kf,0,CV_REDUCE_SUM);
    divSpectrums(kf,den+params.lambda,responsef,cv::DFT_ROWS,false);
    cv::dft(responsef,response,cv::DFT_INVERSE | cv::DFT_ROWS);

    cv::Point2i maxPoint;
    double maxval;
    cv::minMaxLoc(response,NULL,&maxval,NULL,&maxPoint);

    currentScaleFactor = currentScaleFactor * scaleFactors[maxPoint.x];

    if(currentScaleFactor > maxScale){
        currentScaleFactor = maxScale;
    }else if(currentScaleFactor < minScale){
        currentScaleFactor = minScale;
    }
    target.size = cv::Size(round((float)initialSize.width*currentScaleFactor),
     round((float)initialSize.height*currentScaleFactor));
}

void ScaleFilter::learn(const cv::Mat image, const TrackedRegion target){

    // no update
    if(params.learningRate == 0){
        return;
    }

    for(int scale=0; scale < nScales; scale++){
        float scaleFactor = scaleFactors[scale];

        TrackedRegion scaledRegion = TrackedRegion(target.center,initialSize);
        scaledRegion.resize(scaleFactor*currentScaleFactor);

        scaledRegion = scaledRegion.resize(currentScaleFactor*scaleFactor);

        cv::Mat subimage = extractTrackedRegion(image,scaledRegion);
        //resize it to be the same size as the filter
        cv::resize(subimage, scaleSample_, filterSize);
        float windowFactor = window[scale];
        emplaceFeature(scaleSample_,scaleFeatureData_,scale,windowFactor,params.features);
    }

    cv::Mat xf;
    cv::dft(scaleFeatureData_,xf,cv::DFT_ROWS);
    cv::Mat new_den_split,new_den;
    cv::Mat new_num(xf.rows,xf.cols,CV_32FC1);

    cv::mulSpectrums(xf,xf,new_den_split,cv::DFT_ROWS,true);
    cv::reduce(new_den_split,new_den,0,CV_REDUCE_SUM);

    //typ bsxfun
    for(int i=0; i < xf.rows; i++){
        mulSpectrums(yf,xf.row(i),new_num.row(i),cv::DFT_ROWS,true);
    }

    num = (1.0 - params.learningRate)*num + params.learningRate*new_num;
    den = (1.0 - params.learningRate)*den + params.learningRate*new_den;
}

void VisualTracker::addFilter(ACF* f){
    filters.push_back(f);
}

void VisualTracker::track(const cv::Mat currentImage){
    for(std::vector<ACF*>::iterator f = filters.begin(); f != filters.end(); ++f){
        (*f)->track(currentImage,target);
    }
}
void VisualTracker::learn(const cv::Mat currentImage){
    for(std::vector<ACF*>::iterator f = filters.begin(); f != filters.end(); ++f){
        (*f)->learn(currentImage,target);
    }
}

cv::Rect VisualTracker::getBoundingBox(void) const{
    return target.Rect();
}

void VisualTracker::startTracker(const cv::Mat image, const cv::Rect initial_bb){

    target = TrackedRegion(initial_bb);

    for(std::vector<ACF*>::iterator f = filters.begin(); f != filters.end(); ++f){
        (*f)->initialize(image,target);
    }

}

void VisualTracker::reset(const cv::Mat image, const cv::Rect initial_bb){
    for(std::vector<ACF*>::iterator f = filters.begin(); f != filters.end(); ++f){
        (*f)->initialize(image,initial_bb);
    }
}



}