#pragma once

class ImageFeature
{
public:
	ImageFeature(void);
	ImageFeature(const int &threshold);
	ImageFeature(const ImageFeature &image_feature);

	virtual ~ImageFeature(void);

	bool empty() const;
	void clear();
	void copy_to(ImageFeature &dst) const;

	int threshold() const;
	void threshold(const int &val);

	cv::Mat image() const;
	void image(cv::Mat image);

	cv::Mat debug_image();

	std::vector<cv::KeyPoint> keypoints() const;
	cv::Mat descriptors() const;

	std::vector<cv::Point2f> pts() const;

	bool load(const char *filename);
	
	bool process();

	cv::Mat find_homography(const ImageFeature &dst, const double &min_distance_threshold = 0.15, const double &good_matches_distance_threshold = 2.5, const int &matches_count_threshold = 10);

	void draw_keypoints(cv::Mat &canvas);

	static void draw_keypoints(cv::Mat &canvas, std::vector<cv::KeyPoint> &keypoints);

protected:
	int threshold_;

	cv::Mat image_;
	std::vector<cv::KeyPoint> keypoints_;
	cv::Mat descriptors_;

};

