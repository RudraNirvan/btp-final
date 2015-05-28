#include "cv.h"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "ml.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <ctype.h>
#include <windows.h>
#include "wtypes.h"

using namespace std;
using namespace cv;

int main() {
	Mat frame(128, 128, CV_8UC1);
	frame = Scalar(255);
	
	int klass = 2;
	vector<int> compressionParams;
	compressionParams.push_back(CV_IMWRITE_PXM_BINARY);
	
	for(int i=0; i<10; i++) {
		stringstream ss;
		ss<<klass<<"0";
		ss<<i<<".pbm";
		string file = ss.str();
		frame = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		
		for(int j=10; j<100; j+=10) {
			stringstream s;
			s<<klass<<(i + j)<<".pbm";
			cout<<s.str()<<endl;

			imwrite(s.str(), frame, compressionParams);
		} 
//		src = imread(src, )
	}
    return 0;
}
