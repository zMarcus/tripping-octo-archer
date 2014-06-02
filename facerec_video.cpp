/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cstring>
#include <istream>


using namespace cv;
using namespace std;

/*! Main program for the Eigenface Face Recognition */

class EigenFace 
{
public:
    EigenFace();
    void AddNewPerson();
    void FaceDetect();

    
};

EigenFace::EigenFace()
{
	/*! Contructor for a new EigenFace object */
    
}

static void read_csv( const string& filename, vector<Mat>& images, vector<int>& labels, vector<string>& name, char separator = ';')
    {
    ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel,names;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel, separator);
        getline(liness, names);
        if(!path.empty() && !classlabel.empty())
        {
           images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
            name.push_back(names);
        }
    }
    }

void EigenFace::AddNewPerson()
{
	/*! Code for adding in a new person.  This function takes in a person's name, 
	captures 10 pictures of them, resizes and stores those pictures, then finally trains 
	the faces and returns to the main FaceDetect loop.  */

    vector<Mat> images;
    vector<int>labels;
    string fn_csv = "img/imagedatabase.csv";
    ifstream file("img/imagedatabase.csv");
    string line, path, classlabel;
    char separator = ';' ;
    while (getline(file, line))
    {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
    }
        {
         char x;
         strcpy(&x, classlabel.c_str());
         char y = x + 1 ;

         namedWindow("CameraCapture");
         char path[255];
         string name;
         char num[10];
         char jpg[10] = ".jpg";
         int counter = 0;
         cout<<"name" <<endl;
         cin>>name;
        
         CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
         ofstream myfile;
         myfile.open("img/imagedatabase.csv",ios::app);
         while (true)
         {
             
             IplImage* frame = cvQueryFrame( capture );
            
             if ( frame) {
                 Mat original = frame;
                 string fn_haar = "haarcascade_frontalface_default.xml";

                 Mat gray;
                 cvtColor(original,gray, CV_RGB2RGBA);
                 
                 // Find the faces in the frame:
                 vector< Rect_<int> > faces;
                 CascadeClassifier haar_cascade;
                 haar_cascade.load(fn_haar);
                 haar_cascade.detectMultiScale(gray,faces);
                
                 
                 for(int i = 0; i < faces.size(); i++) {
                     // Process face by face:
                     Rect face_i = faces[i];
                     // Crop the face from the image. So simple with OpenCV C++:
                     Mat face = gray(face_i);
                     rectangle(original, face_i, CV_RGB(0, 255,0), 1);
                     imshow( "mywindow", original );

                     // To save 30 images, changed 30 to 10
                     if (counter < 10) {
                         strcpy(path,("C:/Users/003836481/Documents/Visual Studio 2010/projects/openCV_facerec/x64/release/img/" + name).c_str()); //name of person in picture
                         sprintf(num, "%03i", counter);
                         strcat(path, num);
                         strcat(path, jpg);
                         printf("Saving: %s\n", path);
                         
                         // CSV file save
                         myfile << path <<";" << y <<";" << name << endl;

                         imwrite(path, face);
                         Mat nimages = imread(path);
                         Mat sizedimage;
                         
                         Size size (100,100);
                         resize(nimages, sizedimage, size);
                         imwrite(path, sizedimage);
                        
                         counter++;
                         waitKey(500);
                     }

                     if (counter == 30)
                     {
                         cvDestroyWindow("mywindow");
						 cvDestroyWindow("CameraCapture");
                         cvReleaseImage(&frame);
                         FaceDetect();
                     }                
                 }
             }

             waitKey(33);
         }

         myfile.close();
     }
 }




void EigenFace::FaceDetect()
{
    /*! Handles most of the face detection and recognition code.  This is where most of the program operates in. */

    // Check for valid command line arguments, print usage
    // if no arguments were given.
    
	cout << "Test test" << endl;
    
    // Get the path to your CSV:
    string fn_haar = "haarcascade_frontalface_default.xml";
    string fn_csv = "img/imagedatabase.csv";
    
    int deviceId = 0;
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int>labels;
    vector<string>name;
    
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {

        read_csv(fn_csv, images, labels, name);

    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

	if( images.empty() ) {
		cout << "Error occured, image not read correctly" << endl;
	}

    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	cout << "Created Eigenface model" << endl;
    model->train(images, labels);
	cout << "Training complete!" << endl;

    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade included in the project
	// file tree.
	//
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    // Get a handle to the Video device:
    VideoCapture cap(deviceId);

    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return  ;
    }

    // Holds the current frame from the Video device:
    Mat frame;
    
    
    for(;;) {
        
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        namedWindow("");
		
        cvtColor(original,gray, CV_RGB2GRAY);
        
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);

        // At this point you have the position of the faces in
        // vector faces. Now we'll get the faces, make a prediction
        // and annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {

            // Process face by face:
            Rect face_i = faces[i];

            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);

            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            // 
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
			//
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            double predicted_confidence;
            int predicted;
            model -> predict(face_resized,predicted, predicted_confidence);

			// Create variables used for face prediction
            vector<Mat> images;
            vector<int>labels;
            string names;
            ifstream file("img/imagedatabase.csv");
            string line, path, classlabel, pred;
            char separator = ';' ;
			stringstream newpred;
			newpred << predicted;
			pred = newpred.str();

			
			// Perform prediction based on the label in our CSV
            while (getline(file, line))
            {
                stringstream liness(line);
                getline(liness, path, separator);
                getline(liness, classlabel,separator);

                if (pred == classlabel)
                {
                    getline(liness, names);

                    char *cstr = new char[names.length() + 1];
                    strcpy(cstr, names.c_str());    
            
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);

            // Create the text we will annotate the box with:
            string result_message  = format("Persons Name = %s", cstr);
			string confidence_message = format("Confidence = %.0f", predicted_confidence);

            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            
            // And now put it into the image:
            putText(original, result_message, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1, CV_RGB(0,255,0), 2);
            putText(original, confidence_message, Point(pos_x + 10, pos_y + 20), FONT_HERSHEY_PLAIN, 1, CV_RGB(0,255,0), 2);
                }
            }
          }
       
        // And display it:by ace.";
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27){
			break;
		} else if (key == 116){
			AddNewPerson();
		}

		string training_text = "Press 't' to train new face, 'Esc' to exit.";
		putText(original, training_text, Point(20, 20), FONT_HERSHEY_PLAIN, 1, CV_RGB(0,255,0), 2); 
		// Show the result:
        imshow("face_recognizer", original);
    }
    
}

//--------------------------------------------------------
//--------------------------------------------------------
//	Small main function to call all of our functions	|
//--------------------------------------------------------
//--------------------------------------------------------

int main(int argc, const char *argv[])
{
    EigenFace s;
    s.FaceDetect();
    return 0;
}

