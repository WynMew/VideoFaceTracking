#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "FaceDetect.h"
#include "FaceFeature.h"
#include "FaceRecog.h"
#include "FaceProcess.h"
#include "config.h"
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <string>
#include "dirent.h"

#define LEN 5
#define FaceLEN 5

struct FrameRepository {
	cv::Mat item_buffer[LEN];
	cv::Mat IR_buffer[LEN];
	size_t read_position;
	size_t write_position;
	std::mutex mtx;
	std::condition_variable repo_not_full;
	std::condition_variable repo_not_empty;
};

struct FaceRepository {
	cv::Mat item_buffer[LEN];
	cv::Mat IR_buffer[LEN];
	size_t read_position;
	size_t write_position;
	std::mutex mtx;
	std::condition_variable repo_not_full;
	std::condition_variable repo_not_empty;
};


class VideoFaceDetector
{
public:
	VideoFaceDetector(int &capRGB, int &capIR);
	//~VideoFaceDetector();

	FrameRepository gFrameRepository;
	FaceRepository gFaceRepository;
	cv::Mat mFrameNow;

	std::thread VideoCaptureThread();
	std::thread FaceProposerThread();
	std::thread LocalVerifierThread();
	std::thread FaceTrackerThread();

private:
	void InitCamera(int &capRGB, int &capIR);
	void InitFrameRepository(FrameRepository *ir);
	void InitFaceRepository(FaceRepository *ir);
	void GetFrame(); //  Synchronized RGB & IR frame producer
	void ProduceFrameItem(FrameRepository *ir, cv::Mat item, cv::Mat IRimg);
	std::tuple<cv::Mat, cv::Mat> GetFrameItem(FrameRepository *ir);
	double overlap(FR_Rect rectA, FR_Rect rectB);
	void FaceProposer();
	void ProduceFaceItem(FaceRepository *ir, cv::Mat item);
	cv::Mat GetFaceItem(FaceRepository *ir);
	void CreateUserDB();
	void LocalVerifier();
	void FaceTracker();
	int GetBestFace(FR_Rect Rect[], int &iFaceNum);

	int iFrameRow;
	int iFrameCol;
	int iFPS;
	cv::VideoCapture vRGB;
	cv::VideoCapture vIR;

	void * FD_handle = NULL;
	void * FF_handle = NULL;
	int isSearch;

	bool bFaceinTrack;
	float fSearchRatio;

	FaceFeatureDict UserDatabase;
	FaceFeatureDictItem FaceDetectRecogResult;
	FR_Rect EnlargedRect(FR_Rect &InitRect, float &ratio);
};
