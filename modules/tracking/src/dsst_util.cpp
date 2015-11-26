#include "dsst_util.hpp"

namespace cv{

    cv::Mat padSpectrum(const cv::Mat& spectrum, const cv::Size targetSize){
      cv::Mat padded_spectrum = cv::Mat::zeros(targetSize,spectrum.type());

      int cx,cy;
      if(targetSize.width > spectrum.cols){
        cx = spectrum.cols/2;
    }else{
        cx = targetSize.width/2;
    }

    if(targetSize.height > spectrum.rows){
        cy = spectrum.rows/2;
    }else{
        cy = targetSize.height/2;
    }

    cv::Mat q0(spectrum,cv::Rect(0,0,cx,cy));
    cv::Mat q1(spectrum,cv::Rect(spectrum.cols - cx,0,cx,cy));
    cv::Mat q2(spectrum,cv::Rect(0,spectrum.rows - cy,cx,cy));
    cv::Mat q3(spectrum,cv::Rect(spectrum.cols - cx,spectrum.rows - cy,cx,cy));

  //copy the spectrum quadrants into the new matrix
    q0.copyTo(padded_spectrum(cv::Rect(0,0,cx,cy)));
    q1.copyTo(padded_spectrum(cv::Rect(padded_spectrum.cols-cx,0,cx,cy)));
    q2.copyTo(padded_spectrum(cv::Rect(0,padded_spectrum.rows-cy,cx,cy)));
    q3.copyTo(padded_spectrum(cv::Rect(padded_spectrum.cols-cx,padded_spectrum.rows-cy,cx,cy)));

    return padded_spectrum;
}

int totalFeatureDim(const std::vector<TrackerDSST::Feature> featureVec){
    int featureDim = 0;

    for(std::vector<TrackerDSST::Feature>::const_iterator f = featureVec.begin();
        f != featureVec.end(); ++f){
        switch(*f){
            case TrackerDSST::Gray:
            featureDim += 1;
            break;
            case TrackerDSST::CN:
            featureDim += 11;
            break;
            case TrackerDSST::HOG:
            featureDim += 31;
            break;
            case TrackerDSST::FHOG4I:
            featureDim += 32;
            break;
            case TrackerDSST::FHOG1:
            featureDim += 31;
            break;
            case TrackerDSST::FHOG4:
            featureDim += 31;
            break;
            default:
                //throw liuexception("Invalid feature in totalFeatureDim");
            break;
        }
    }
    return featureDim;
}

cv::Mat extractTrackedRegion(const cv::Mat image, const TrackedRegion region){

    int xMin = region.center.x - floor(((float)region.size.width)/2.0);
    int yMin = region.center.y - floor(((float)region.size.height)/2.0);

    //int xMax = region.center.x + ceil(((float)region.size.width)/2.0);
    //int yMax = region.center.y + ceil(((float)region.size.height)/2.0);

    int xMax = xMin + region.size.width;
    int yMax = yMin + region.size.height;

    int xMinPad,xMaxPad,yMinPad,yMaxPad;

    if(xMin < 0){
        xMinPad = -xMin;
    }else{
        xMinPad = 0;
    }

    if(xMax > image.size().width){
        xMaxPad = xMax - image.size().width;
    }else{
        xMaxPad = 0;
    }

    if(yMin < 0){
        yMinPad = -yMin;
    }else{
        yMinPad = 0;
    }

    if(yMax > image.size().height){
        yMaxPad = yMax - image.size().height;
    }else{
        yMaxPad = 0;
    }

    //compute the acual rectangle we will extract from the image
    cv::Rect extractionRegion = cv::Rect(xMin + xMinPad,
     yMin + yMinPad,
     (xMax-xMin) - xMaxPad - xMinPad,
     (yMax-yMin) - yMaxPad - yMinPad);


    //make sure the patch is not completely outside the image
    if(extractionRegion.x + extractionRegion.width > 0 &&
       extractionRegion.y + extractionRegion.height > 0 &&
       extractionRegion.x < image.cols &&
       extractionRegion.y < image.rows){


        cv::Mat real_patch(region.size,image.type());


        //replicate along borders if needed
    if(xMinPad > 0 || xMaxPad > 0 || yMinPad > 0 || yMaxPad > 0){
        cv::copyMakeBorder(image(extractionRegion), real_patch, yMinPad,
           yMaxPad, xMinPad, xMaxPad, cv::BORDER_REPLICATE);

    }else{
        real_patch = image(extractionRegion);
    }

    if(!(real_patch.size().width == region.size.width && real_patch.size().height == region.size.height)){
            //cout << "kasst" << endl;
    }

    return real_patch;

}else{
    cv::Mat dummyRegion = cv::Mat::zeros(region.size,image.type());
    return dummyRegion;
}
}
/*
*  Assumes the feature matrix is correctly initialized and places the feature on the suggested dimension
*  along with the correct scaling, and possible feature weighting
*/
void emplaceFeature(cv::Mat& imagePatch, cv::Mat& featureMatrix, const int col, const float featureWeight, const std::vector<TrackerDSST::Feature> featureVec){


    //TODO this is very suboptimally implemented, make it faster
    if(featureVec[0] == TrackerDSST::Gray){
        cv::Mat grayImage;
        cv::Mat featureColumn = featureMatrix.col(col);
        cvtColor(imagePatch,grayImage,CV_BGR2GRAY);
        grayImage = grayImage.reshape(0,grayImage.size().area());
        grayImage.convertTo(featureColumn,CV_32FC1,((1.0/255.0)*featureWeight),-0.5);
    }else if(featureVec[0] == TrackerDSST::FHOG1){
        std::vector<cv::Mat> hogVector;
        getFhog(imagePatch, hogVector, 1);

        //for(auto dataVector : hogVector){
        for(std::vector<cv::Mat>::iterator dataVector = hogVector.begin();
            dataVector != hogVector.end(); ++dataVector){

            cv::Mat dataCol = ( (*dataVector).reshape(0,(*dataVector).size().area())) * featureWeight;
        cv::Mat outputColumn = featureMatrix.col(col);

        for(int r = 0; r < featureMatrix.rows; r++){
            featureMatrix.at<float>(r,col) = dataCol.at<float>(0,r);
        }
    }
}else if(featureVec[0] == TrackerDSST::FHOG4){

    std::vector<cv::Mat> hogVector;
    getFhog(imagePatch, hogVector, 4);

    for(size_t d=0; d < hogVector.size(); d++){
        int elementPerDim = hogVector[d].size().area();
        cv::Mat dataCol = hogVector[d].reshape(0,elementPerDim);

        for(int r = 0; r < elementPerDim; r++){
            featureMatrix.at<float>(d*elementPerDim + r,col) = dataCol.at<float>(r,0);
        }
            //cv::Mat dataCol = (dataVector.reshape(0,dataVector.size().area())) * featureWeight;

    }
}
    //cout << featureMatrix << endl;
    //cv::waitKey(0);
}


std::vector<cv::Mat> extractFeatures(cv::Mat& image_patch, const std::vector<TrackerDSST::Feature> featureVec){
    std::vector<cv::Mat> featureVector;

    for(std::vector<TrackerDSST::Feature>::const_iterator feature = featureVec.begin();
        feature != featureVec.end(); ++feature){

        switch(*feature){
            case TrackerDSST::Gray:{
                cv::Mat grayImage;
                cv::Mat grayFloat;
                cvtColor(image_patch,grayImage,CV_BGR2GRAY);
                grayImage.convertTo(grayFloat,CV_32FC1);
                featureVector.push_back(grayFloat/255.0);
            }
            break;
            case TrackerDSST::CN:{
                std::vector<cv::Mat> cnFeature = getColornames(image_patch);
                featureVector.insert(featureVector.end(),cnFeature.begin(),cnFeature.end());
            }
            break;
            case TrackerDSST::HOG:{
                std::vector<cv::Mat> hogFeatures;
                getFhog(image_patch,hogFeatures,4);
                //std::vector<cv::Mat> hogFeatures = extractHOGfeature(image_patch);
                featureVector.insert(featureVector.end(),hogFeatures.begin(),hogFeatures.end());
            }
            break;
            case TrackerDSST::FHOG4I:{
                std::vector<cv::Mat> hogFeatures;
                getFhog(image_patch,hogFeatures,4);
                //std::vector<cv::Mat> hogFeatures = extractHOGfeature(image_patch);
                featureVector.insert(featureVector.end(),hogFeatures.begin(),hogFeatures.end());
                cv::Mat grayImage,grayFloatImage;
                cvtColor(image_patch,grayImage,CV_BGR2GRAY);
                grayImage.convertTo(grayFloatImage,CV_32FC1,1.0/255.0,-0.5);
                cv::Mat blockImage = blockAverageImage(grayFloatImage,4);
                featureVector.push_back(blockImage);
            }
            break;
            default:
            break;
        }
    }

    return featureVector;
}


//the multiplication/sum part of the ACF filter algorithm
cv::Mat channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb){
    CV_Assert(a.size() == b.size());
    CV_Assert(a.size() > 1);

    cv::Mat prod;
    cv::Mat sum = cv::Mat::zeros(a[0].size(),a[0].type());
    for(unsigned int i = 0; i < a.size(); ++i){
        cv::Mat ca = a[i];
        cv::Mat cb = b[i];
        cv::mulSpectrums(a[i],b[i],prod,flags,conjb);
        sum += prod;
    }
    return sum;
}


void divSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB){
    cv::Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    cv::Mat dst = _dst.getMat();

    bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
     srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if( is_1d && !(flags & cv::DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataA = (const float*)srcA.data;
        const float* dataB = (const float*)srcB.data;
        float* dataC = (float*)dst.data;
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                        (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] +
                        (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] -
                        (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                    else
                        for( j = 1; j <= rows - 2; j += 2 )
                        {

                            double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                            (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                            double re = (double)dataA[j*stepA]*dataB[j*stepB] -
                            (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                            double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] +
                            (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                            dataC[j*stepC] = (float)(re / denom);
                            dataC[(j+1)*stepC] = (float)(im / denom);
                        }
                        if( k == 1 )
                            dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
                    }
                }

                for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
                {
                    if( is_1d && cn == 1 )
                    {
                        dataC[0] = dataA[0] / (dataB[0] + eps);
                        if( cols % 2 == 0 )
                            dataC[j1] = dataA[j1] / (dataB[j1] + eps);
                    }

                    if( !conjB )
                        for( j = j0; j < j1; j += 2 )
                        {
                            double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                            double re = (double)(dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1]);
                            double im = (double)(dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1]);
                            dataC[j] = (float)(re / denom);
                            dataC[j+1] = (float)(im / denom);
                        }
                        else
                            for( j = j0; j < j1; j += 2 )
                            {
                                double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                                double re = (double)(dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1]);
                                double im = (double)(dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1]);
                                dataC[j] = (float)(re / denom);
                                dataC[j+1] = (float)(im / denom);
                            }
                        }
                    }
                    else
                    {
                        const double* dataA = (const double*)srcA.data;
                        const double* dataB = (const double*)srcB.data;
                        double* dataC = (double*)dst.data;
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                        dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] +
                        dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] -
                        dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                    else
                        for( j = 1; j <= rows - 2; j += 2 )
                        {
                            double denom = dataB[j*stepB]*dataB[j*stepB] +
                            dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                            double re = dataA[j*stepA]*dataB[j*stepB] -
                            dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                            double im = dataA[(j+1)*stepA]*dataB[j*stepB] +
                            dataA[j*stepA]*dataB[(j+1)*stepB];

                            dataC[j*stepC] = re / denom;
                            dataC[(j+1)*stepC] = im / denom;
                        }
                        if( k == 1 )
                            dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
                    }
                }

                for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
                {
                    if( is_1d && cn == 1 )
                    {
                        dataC[0] = dataA[0] / (dataB[0] + eps);
                        if( cols % 2 == 0 )
                            dataC[j1] = dataA[j1] / (dataB[j1] + eps);
                    }

                    if( !conjB )
                        for( j = j0; j < j1; j += 2 )
                        {
                            double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                            double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
                            double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
                            dataC[j] = re / denom;
                            dataC[j+1] = im / denom;
                        }
                        else
                            for( j = j0; j < j1; j += 2 )
                            {
                                double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                                double re = dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1];
                                double im = dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1];
                                dataC[j] = re / denom;
                                dataC[j+1] = im / denom;
                            }
                        }
                    }
                }


                inline float getColorname(int cname,cv::Vec3b pixelValue){
                    int x = pixelValue(0) >> 3;
                    int y = pixelValue(1) >> 3;
                    int z = pixelValue(3) >> 3;
                    int i = cname * 32 * 32 * 32 + z * 32*32 + y*32 + x;
                    return cn_data[i];
                }

                std::vector<cv::Mat> getColornames(cv::Mat image){
                    std::vector<cv::Mat> colorchannels(11);
                    float* p;
                    float norm_term = -1.0/11.0;
                    for(int cname = 0; cname < 11; ++cname){
                        colorchannels[cname] = cv::Mat(image.size(),CV_32FC1);

                        for(int x=0; x < image.rows; ++x){
                            p = colorchannels[cname].ptr<float>(x);

                            for(int y=0; y < image.cols; ++y){
                                p[y] = getColorname(cname,image.at<cv::Vec3b>(x,y)) - norm_term;
                            }
                        }
                    }
                    return colorchannels;
                }

                void getFhog(const cv::Mat& inputImage, std::vector<cv::Mat>& featureVector, const int cell_size){
    /*
    dlib::cv_image<dlib::bgr_pixel> img(inputImage);

    dlib::array<dlib::array2d<float>> fhog_image;
    dlib::extract_fhog_features(img,fhog_image,cell_size);

    for(int i=0; i < fhog_image.size(); i++){
        cv::Mat layer = dlib::toMat(fhog_image[i]);
        featureVector.push_back(layer.clone());
    }
    */
}

cv::Mat blockAverageImage(const cv::Mat& inputImage, const int cell_size){

    int rows = round(inputImage.rows/ (float)cell_size)-2;
    int cols = round(inputImage.cols/ (float)cell_size)-2;

    cv::Mat blockAverage = cv::Mat::zeros(rows,cols,CV_32FC1);

    for(int r = 0; r < rows*4; r++){
        const float* readPtr = inputImage.ptr<float>(r);

        int cell_row = floor(r/cell_size);
        float* writePtr = blockAverage.ptr<float>(cell_row);

        for(int c = 0; c < cols*4; c++){
            int cell_col = floor(c/cell_size);
            writePtr[cell_col] += readPtr[c];
        }
    }

    return (blockAverage / (float)cell_size);
}

}