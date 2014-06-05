#include "stdafx.h"
#include "ImageFeature.h"

std::string target_filename = "target.png";

int main(int argc, char* argv[])
{
	// open marker image
	ImageFeature target_feature(3000);
	if (target_feature.load(target_filename.c_str()) == false) {
		std::cerr << "cannot open imagefile...filename=" << target_filename << std::endl;
		return -1;
	}

	// open capture device
	cv::Mat capture_img;
	ImageFeature capture_feature(1000);
	cv::VideoCapture capture;
	capture.open(0);

	cv::Mat canvas_img;

	// main loop
	while(true) {
		capture >> capture_img;
		capture_feature.image(capture_img);

		capture_feature.debug_image().copyTo(canvas_img);

		cv::Mat h_mat = target_feature.find_homography(capture_feature);

		if (!h_mat.empty()) {
			std::vector<cv::Point2f> dst_ps;
			cv::perspectiveTransform(target_feature.pts(), dst_ps, h_mat);
			for (unsigned int i = 0; i < dst_ps.size(); ++i) {
				cv::line(canvas_img, dst_ps[i], dst_ps[(i+1) % dst_ps.size()], CV_RGB(0,255,0), 2);
			}
		}

		cv::imshow("marker", target_feature.debug_image());
		cv::imshow("capture_img", canvas_img);

		int c = cv::waitKey(1);  
		if (c == 27) break;
	}

	capture.release();
	cv::destroyAllWindows();
}

