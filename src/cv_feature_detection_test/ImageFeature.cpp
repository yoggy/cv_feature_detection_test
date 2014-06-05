#include "StdAfx.h"
#include "ImageFeature.h"
#include <algorithm>
#include <iterator> 

ImageFeature::ImageFeature(void) : threshold_(2000)
{
}

ImageFeature::ImageFeature(const int &threshold) : threshold_(threshold)
{
	
}

ImageFeature::ImageFeature(const ImageFeature &image_feature)
{
	image_feature.copy_to(*this);
}

ImageFeature::~ImageFeature(void)
{
	clear();
}

bool ImageFeature::empty() const
{
	return image_.empty();
}

void ImageFeature::clear()
{
	image_.release();
	keypoints_.clear();
	descriptors_.release();
}

void ImageFeature::copy_to(ImageFeature &dst) const
{
	this->image_.copyTo(dst.image_);

	dst.keypoints_.clear();
	std::copy(this->keypoints_.begin(), this->keypoints_.end(), std::back_inserter(dst.keypoints_));

	this->descriptors_.copyTo(dst.descriptors_);
}

int ImageFeature::threshold() const
{
	return threshold_;
}

void ImageFeature::threshold(const int &val)
{
	threshold_ = val;
	process();
}

cv::Mat ImageFeature::image() const
{
	return image_;
}

void ImageFeature::image(cv::Mat image)
{
	image.copyTo(image_);
	process();
}

cv::Mat ImageFeature::debug_image()
{
	if (empty()) return cv::Mat();

	cv::Mat canvas;
	image_.copyTo(canvas);	

	canvas.create(image_.size(), CV_8UC3);
	draw_keypoints(canvas, keypoints_);

	return canvas;
}

std::vector<cv::KeyPoint> ImageFeature::keypoints() const
{
	return keypoints_;
}

cv::Mat ImageFeature::descriptors() const
{
	return descriptors_;
}

bool ImageFeature::load(const char *filename)
{
	image_.release();
	image_ = cv::imread(filename, 1);
	if (image_.empty()) return false;

	process();

	return true;
}

std::vector<cv::Point2f> ImageFeature::pts() const 
{
	std::vector<cv::Point2f> pts;

	if (empty()) {
		return pts;
	}

	// counter clockwise
	pts.push_back(cv::Point2f(0.0f, 0.0f));
	pts.push_back(cv::Point2f(image_.cols - 1.0f, 0.0f));
	pts.push_back(cv::Point2f(image_.cols - 1.0f, image_.rows - 1.0f));
	pts.push_back(cv::Point2f(0.0f, image_.rows - 1.0f));

	return pts;
}

bool ImageFeature::process()
{
	if (empty()) return false;

	cv::Mat gray_img;
	cv::cvtColor(image_, gray_img, CV_BGR2GRAY);
	cv::normalize(gray_img, gray_img, 0, 255, cv::NORM_MINMAX);

	cv::SurfFeatureDetector detector(threshold_);
	detector.detect(gray_img, keypoints_);

	cv::SurfDescriptorExtractor extractor;
	extractor.compute(image_, keypoints_, descriptors_);

	return true;
}

cv::Mat ImageFeature::find_homography(const ImageFeature &dst, const double &min_distance_threshold, const double &matches_distance_threshold, const int &matches_count_threshold)
{
	cv::Mat h;

	std::vector<cv::DMatch> matches;
	std::vector<cv::DMatch> good_matches;
	std::vector<cv::DMatch>::iterator it;

	cv::BFMatcher matcher(cv::NORM_L2);

	matcher.match(this->descriptors(), dst.descriptors(), matches);

	double min_d = DBL_MAX;
	for (it = matches.begin(); it != matches.end(); ++it) {
	    double d = it->distance;
	    if (d < min_d) min_d = d;
	}

	if (min_d > min_distance_threshold) return cv::Mat();
 
	for (it = matches.begin(); it != matches.end(); ++it) {
		if (it->distance <= min_d * matches_distance_threshold) good_matches.push_back(*it);
	}

	if (good_matches.size() < (unsigned int)matches_count_threshold) return cv::Mat();

	std::vector<cv::Point2f> src_ps;
	std::vector<cv::Point2f> dst_ps;
	for (it = good_matches.begin(); it != good_matches.end(); ++it) {
		src_ps.push_back(this->keypoints()[it->queryIdx].pt);		
		dst_ps.push_back(dst.keypoints()[it->trainIdx].pt);		
	}

	h = cv::findHomography(src_ps, dst_ps, cv::RANSAC);

	return h;
}

void ImageFeature::draw_keypoints(cv::Mat &canvas)
{
	draw_keypoints(canvas, keypoints_);
}

void ImageFeature::draw_keypoints(cv::Mat &canvas, std::vector<cv::KeyPoint> &keypoints)
{
	std::vector<cv::KeyPoint>::iterator it;

	for(it = keypoints.begin(); it != keypoints.end(); ++it) {
		cv::circle(canvas, it->pt, (int)(it->size / 3), CV_RGB(0, 0, 255), 1, CV_AA);

		cv::Point pt((int)(it->pt.x + cos(it->angle)*it->size / 3), (int)(it->pt.y + sin(it->angle)*it->size / 3));
		cv::line(canvas, it->pt, pt, CV_RGB(0, 255, 0), 2, CV_AA);

		cv::circle(canvas, it->pt, 2, CV_RGB(255, 0, 0), -1);
	}
}
