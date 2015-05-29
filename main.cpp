#include "cv.h"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "ml.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <ctype.h>
#include <time.h>
#include <windows.h>
#include "wtypes.h"

using namespace std;
using namespace cv;

/****************************** preprocessing.cpp *******************/
/*****************************************************************
*
* Find the min box. The min box respect original aspect ratio image 
* The image is a binary data and background is white.
*
*******************************************************************/
void findX(IplImage* imgSrc,int* min, int* max){
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min 
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->width; i++){
		cvGetCol(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0]){
			*max= i;
			if(!minFound){
				*min= i;
				minFound= 1;
			}
		}
	}
}

void findY(IplImage* imgSrc,int* min, int* max){
	int i;
	int minFound=0;
	CvMat data;
	CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
	CvScalar val=cvRealScalar(0);
	//For each col sum, if sum < width*255 then we find the min 
	//then continue to end to search the max, if sum< width*255 then is new max
	for (i=0; i< imgSrc->height; i++){
		cvGetRow(imgSrc, &data, i);
		val= cvSum(&data);
		if(val.val[0] < maxVal.val[0]){
			*max=i;
			if(!minFound){
				*min= i;
				minFound= 1;
			}
		}
	}
}
CvRect findBB(IplImage* imgSrc){
	CvRect aux;
	int xmin, xmax, ymin, ymax;
	xmin=xmax=ymin=ymax=0;

	findX(imgSrc, &xmin, &xmax);
	findY(imgSrc, &ymin, &ymax);
	
	aux=cvRect(xmin, ymin, xmax-xmin, ymax-ymin);
	
	return aux;
	
}

IplImage preprocessing(IplImage* imgSrc,int new_width, int new_height){
	IplImage* result;
	IplImage* scaledResult;

	CvMat data;
	CvMat dataA;
	CvRect bb;//bounding box
	CvRect bba;//boundinb box maintain aspect ratio
	
	//Find bounding box
	bb=findBB(imgSrc);
	
	//Get bounding box data and no with aspect ratio, the x and y can be corrupted
	cvGetSubRect(imgSrc, &data, cvRect(bb.x, bb.y, bb.width, bb.height));
	//Create image with this data with width and height with aspect ratio 1 
	//then we get highest size betwen width and height of our bounding box
	int size=(bb.width>bb.height)?bb.width:bb.height;
	result=cvCreateImage( cvSize( size, size ), 8, 1 );
	cvSet(result,CV_RGB(255,255,255),NULL);
	//Copy de data in center of image
	int x=(int)floor((float)(size-bb.width)/2.0f);
	int y=(int)floor((float)(size-bb.height)/2.0f);
	cvGetSubRect(result, &dataA, cvRect(x,y,bb.width, bb.height));
	cvCopy(&data, &dataA, NULL);
	//Scale result
	scaledResult=cvCreateImage( cvSize( new_width, new_height ), 8, 1 );
	cvResize(result, scaledResult, CV_INTER_NN);
	
	//Return processed data
	return *scaledResult;
	
}


/***************************** basicOCR.cpp ***************************/
class basicOCR{
	public:
		float classify(IplImage* img,int showResult);
		basicOCR ();
		void test();	
	private:
		char file_path[255];
		int train_samples;
		int total_samples;
		int classes;
		CvMat* trainData;
		CvMat* trainClasses;
		int size;
		static const int K=10;
		CvKNearest *knn;
		void getData();
		void train();
};

void basicOCR::getData()
{
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	for(i =0; i<classes; i++){
		for( j = 0; j< train_samples; j++){
			
			//Load file
			if(j<10)
				sprintf(file,"%s%d/%d0%d.pbm",file_path, i, i , j);
			else
				sprintf(file,"%s%d/%d%d.pbm",file_path, i, i , j);
			src_image = cvLoadImage(file,0);
			if(!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			
			//Set class label
			cvGetRow(trainClasses, &row, i*train_samples + j);
			cvSet(&row, cvRealScalar(i));
			//Set data 
			cvGetRow(trainData, &row, i*train_samples + j);

			IplImage* img = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
			//convert 8 bits image to 32 float image
			cvConvertScale(&prs_image, img, 0.0039215, 0);

			cvGetSubRect(img, &data, cvRect(0,0, size,size));
			
			CvMat row_header, *row1;
			//convert data matrix sizexsize to vecor
			row1 = cvReshape( &data, &row_header, 0, 1 );
			cvCopy(row1, &row, NULL);
		}
	}
}

void basicOCR::train()
{
	knn=new CvKNearest( trainData, trainClasses, 0, false, K );
}

float basicOCR::classify(IplImage* img, int showResult)
{
	IplImage prs_image;
	CvMat data;
	CvMat* nearest=cvCreateMat(1,K,CV_32FC1);
	float result;
	//process file
	prs_image = preprocessing(img, size, size);
	
	//Set data 
	IplImage* img32 = cvCreateImage( cvSize( size, size ), IPL_DEPTH_32F, 1 );
	cvConvertScale(&prs_image, img32, 0.0039215, 0);
	cvGetSubRect(img32, &data, cvRect(0,0, size,size));
	CvMat row_header, *row1;
	row1 = cvReshape( &data, &row_header, 0, 1 );

	result=knn->find_nearest(row1,K,0,0,nearest,0);
	
	int accuracy=0;
	for(int i=0;i<K;i++){
		if( nearest->data.fl[i] == result)
                    accuracy++;
	}
	float pre=100*((float)accuracy/(float)K);
	if(showResult==1){
		char key[] = {'S', '6', 'O', '9', '7'};
//		printf("|\t%.0f\t| \t%.2f%%  \t| \t%d of %d \t| \n",result,pre,accuracy,K);
		printf("|\t%.0f = %c\t| \t%.2f%%  \t| \t%d of %d \t| \n", result, key[(int)result], pre, accuracy,K);
		printf(" ---------------------------------------------------------------\n");
		
		string out_s;
		out_s.push_back(key[(int)result]);
		
		Mat output(200, 200, CV_8UC3);
		output = Scalar(255, 255, 255);
		putText(output, out_s, cvPoint(20, 180), FONT_HERSHEY_COMPLEX_SMALL, 12, cvScalar(0, 0, 255), 3, 1);
		imshow("result", output);
	}

	return result;

}

void basicOCR::test(){
	IplImage* src_image;
	IplImage prs_image;
	CvMat row,data;
	char file[255];
	int i,j;
	int error=0;
	int testCount=0;
	for(i =0; i<classes; i++){
		for( j = train_samples; j < total_samples; j++){
			
			sprintf(file,"%s%d/%d0%d.pbm",file_path, i, i , j);
			if(j<10)
				sprintf(file,"%s%d/%d0%d.pbm",file_path, i, i , j);
			else
				sprintf(file,"%s%d/%d%d.pbm",file_path, i, i , j);
				
			src_image = cvLoadImage(file,0);
			if(!src_image){
				printf("Error: Cant load image %s\n", file);
				//exit(-1);
			}
			//process file
			prs_image = preprocessing(src_image, size, size);
			float r=classify(&prs_image,0);
			if((int)r!=i)
				error++;
			
			testCount++;
		}
	}
	float totalerror=100*(float)error/(float)testCount;
	printf("System Error: %.2f%%\n", totalerror);
	
}

basicOCR::basicOCR()
{

	//initial
	sprintf(file_path , "OCR-20/");
	train_samples = 70;
	total_samples = 100;
	classes = 5;
	size = 40;

	trainData = cvCreateMat(train_samples*classes, size*size, CV_32FC1);
	trainClasses = cvCreateMat(train_samples*classes, 1, CV_32FC1);

	//Get data (get images and process it)
	getData();
	
	//train	
	train();
	//Test	
	test();
	
	printf(" ---------------------------------------------------------------\n");
	printf("|\tClass\t|\tPrecision\t|\tAccuracy\t|\n");
	printf(" ---------------------------------------------------------------\n");

	
}

/****************************** main.cpp *********************/
IplImage* imagen;
int red,green,blue;
IplImage* screenBuffer;
int drawing;
int r,last_x, last_y;

void getScreenResolution(int &horizontal, int &vertical) {
	RECT desktop;
	// Get a handle to the desktop window
	const HWND hDesktop = GetDesktopWindow();
	// Get the size of screen to the variable desktop
	GetWindowRect(hDesktop, &desktop);
	// The top left corner will have coordinates (0,0)
	// and the bottom right corner will have coordinates
	// (horizontal, vertical)
	horizontal = desktop.right;
	vertical = desktop.bottom;
}

int main() {
	VideoCapture cap(0);
	
	if(!cap.isOpened()) {
		cout<<"Unable to open webcam.";
		return 1;
	}
	
//	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
//	namedWindow("Original", CV_WINDOW_AUTOSIZE);
//	namedWindow("Black & White", CV_WINDOW_AUTOSIZE);
	
	int screenHorizontal, screenVertical;
	getScreenResolution(screenHorizontal, screenVertical);
	cout<<screenHorizontal<<"x"<<screenVertical<<endl;
	
	/*************Finding a colour's HSV values*********/
//	int hLow, hHigh, sLow, sHigh, vLow, vHigh;
//	hLow = sLow = vLow = 0;
//	hHigh = 179;
//	sHigh = vHigh = 255;
//	
//	createTrackbar("H low", "Webcam", &hLow, 255);
//	createTrackbar("H high", "Webcam", &hHigh, 179);
//	createTrackbar("S low", "Webcam", &sLow, 255);
//	createTrackbar("S high", "Webcam", &sHigh, 255);
//	createTrackbar("V low", "Webcam", &vLow, 255);
//	createTrackbar("V high", "Webcam", &vHigh, 255);
	/**************Finding HSV ends*********************/
	
	double width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	double height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	
	int prevX = -1, prevY = -1;
	
	cout<<width<<"x"<<height<<endl; //640x480
	Mat imm;
	cap.read(imm);
//	Mat imgLines = Mat::ones( imm.size(), CV_8UC3 );;
	Mat imgLines(height, width, CV_8UC1);
	imgLines = Scalar(255);
	
	basicOCR ocr;
	int count = 0;
	
	bool timer = false;
	time_t startTime;
		
	while(1) {		
		Mat frame, temp, ori;
		if(!cap.read(frame)) {
			cout<<"Unable to read frame from webcam.";
			return 1;
		}
		
		flip(frame, frame, 1); // horizontal flipping
		ori = frame;
		
		/************** Segmentation  *********************/
		cvtColor(frame, temp, CV_RGB2HSV);
		frame = temp;
		
		// 109 - 142, 18 - 156, 164 - 255 ---> pink nose
		// 0 - 25, 134 - 255, 51 - 255 ---> blue sharpener
//		inRange(frame, Scalar(hLow, sLow, vLow), Scalar(hHigh, sHigh, vHigh), temp);
		inRange(frame, Scalar(0, 134, 51), Scalar(25, 255, 255), temp);
		frame = temp;
		
//		namedWindow("Noise", CV_WINDOW_AUTOSIZE);
//		imshow("Noise", frame);
		
		
		/*************** Denoise ***********************/
		int size = 3;
		Mat element = getStructuringElement(MORPH_RECT, Size(2*size + 1, 2*size + 1), Point(size, size));
		erode(frame, temp, element);
		
		size = 3;
		element = getStructuringElement(MORPH_RECT, Size(2*size + 1, 2*size + 1), Point(size, size));
		dilate(temp, frame, element);
		
		size = 3;
		element = getStructuringElement(MORPH_RECT, Size(2*size + 1, 2*size + 1), Point(size, size));
		dilate(frame, temp, element);
		
		size = 3;
		element = getStructuringElement(MORPH_RECT, Size(2*size + 1, 2*size + 1), Point(size, size));
		erode(temp, frame, element);
		
		/************************************************/
		Moments mom = moments(frame);
		double m01 = mom.m01;
		double m10 = mom.m10;
		double area = mom.m00;
		
		int x = m10/area;
		int y = m01/area;
		
		if(prevX >= 0 && prevY >= 0 && x >= 0 && y >= 0) {
//			line(imgLines, Point(x, y), Point(prevX, prevY), Scalar(0, 0, 255), 3);
			line(imgLines, Point(x, y), Point(prevX, prevY), Scalar(0), 20);
			int posX = (x/width)*screenHorizontal;
			int posY = (y/height)*screenVertical;

			/********Mouse Control***/
//			SetCursorPos(posX, posY);
			/********Scroll*********/
//			if(y > prevY)
//				mouse_event(MOUSEEVENTF_WHEEL, 0, 0, -30, 0);
//			else if(y < prevY)
//				mouse_event(MOUSEEVENTF_WHEEL, 0, 0, 30, 0);
		}
		
		prevX = x, prevY = y;
		
		imshow("Webcam", frame);
//		imshow("Original", ori+imgLines);
//		imshow("Black & White", imgLines);
		
		char c = waitKey(30);
		if(c == 'r') {
			imgLines = Scalar(255);
		} else if(c == 's') {
			Mat saveImg;
//			cvtColor(imgLines, saveImg, CV_RGB2GRAY);
			resize(imgLines, saveImg, Size(128, 128));
			
			namedWindow("resized", CV_WINDOW_AUTOSIZE);
			imshow("resized", saveImg);
			
			vector<int> compressionParams;
			compressionParams.push_back(CV_IMWRITE_PXM_BINARY);
			
			stringstream ss;
			/*
				1. To save images, first change the number below to the class which you are targetting.
				2. [NOT NEEDED ANYMORE] Change the path below.
			*/
			ss<<2;
			if(count < 10)
				ss<<0<<count;
			else
				ss<<count;
				
//			string save = "C:/Users/Rudra/Documents/Dev C++ Codes/btp final/" + ss.str() + ".pbm";
			string save = ss.str() + ".pbm";
			cout<<save<<endl;
			
			imwrite(save, saveImg, compressionParams);
			count++;
		} else if(c == 'c') {
			Mat img;
			resize(imgLines, img, Size(128, 128));
			namedWindow("resized", CV_WINDOW_AUTOSIZE);
//			imshow("resized", img);
			IplImage copy = img;
			IplImage *new_img = &copy;
			ocr.classify(new_img, 1);
		} else if(c == 27) {
			cout<<"ESC pressed.";
			break;
		} else {
			 // frame == blue-coloured pencil sharpener
			if(countNonZero(frame) == 0) { // no shapener == black image
				if( (imgLines.rows * imgLines.cols) - countNonZero(imgLines) == 0) { // no line drawn == white image
					continue;
				} else { // line drawn
					if(timer == false) {
						timer = true;
						startTime = time(NULL);
						continue;
					} else {
						double difft = difftime(time(NULL), startTime);
						int diff = difft * 1000; // diff == milliseconds
						if(diff < 70) {
							continue;
						}
						
						// lots of time has passed since last blue colour was visible
						timer = false;
					}
					Mat img;
					resize(imgLines, img, Size(128, 128));
//					imshow("resized", img);
					IplImage copy = img;
					IplImage *new_img = &copy;
					ocr.classify(new_img, 1);
					
					// reset image to blank
					imgLines = Scalar(255);
				}
			}
		}
	}


    return 0;
}
