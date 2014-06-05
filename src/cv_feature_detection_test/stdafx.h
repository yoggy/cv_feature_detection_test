#pragma once

#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <stdio.h>
#include <stdlib.h>

// opencv
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#if _DEBUG
	#pragma comment(lib, "opencv_core249d.lib")
	#pragma comment(lib, "opencv_imgproc249d.lib")
	#pragma comment(lib, "opencv_highgui249d.lib")
	#pragma comment(lib, "opencv_features2d249d.lib")
	#pragma comment(lib, "opencv_calib3d249d.lib")
	#pragma comment(lib, "opencv_nonfree249d.lib")
#else
	#pragma comment(lib, "opencv_core249.lib")
	#pragma comment(lib, "opencv_imgproc249.lib")
	#pragma comment(lib, "opencv_highgui249.lib")
	#pragma comment(lib, "opencv_features2d249.lib")
	#pragma comment(lib, "opencv_calib3d249.lib")
	#pragma comment(lib, "opencv_nonfree249.lib")
#endif
