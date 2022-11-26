#include <iostream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// for object detections
#include <opencv2/objdetect.hpp>
#include <map>


using namespace cv;
using namespace std;

const string resourcePath = R"(C:\Code\opencvLearn\Resources\)";
#define ESC_KEY 27
#define Q_KEY 113

void testvideo();
void testwebcam();
void task1(); // Image manipulation (convert to grayscale, blur, detect edges, dilate, erode)
void task2(); // Image resize and crop
void task3(); // Drawing to image
void task4(); // Image warping
void task5(); // Color detection (masking, HSV selectors)
void task6(); // Shape detection
void task7(); // Face detection
void task8(); // Document scanner (image)
void task9(); // Number plate detection

Mat loadimg(const string&);

Mat getWarp(Mat mat, vector<Point> vector1, float w, float h);

// image manipulation
void task1() {
    auto img = loadimg("test.png");
    Mat imgGray;

    // Convert image color repr BGR -> GRAYSCALE
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    // Blur
    GaussianBlur(imgGray, imgGray, Size(127,127), 1, 1);

    // Edge detection
    Canny(imgGray, imgGray, 25, 75);

    // Fixing edge detection
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgGray, imgGray, kernel);
    erode(imgGray, imgGray, kernel);

    imshow("Gray", imgGray);
    waitKey(0);
}

// Resize and crop
void task2() {
    Mat
    imgResize
    ,imgCrop
    ;

    auto img = loadimg("test.png");

    // Resizing image
    resize(img, imgResize, Size(), .5, .5);

    // Cropping image
    imgCrop = img(Rect(200,50,300,350));


    imshow("Gray", img);
    imshow("Resized", imgResize);
    imshow("Cropped", imgCrop);
    waitKey(0);

}

// Drawing to the image
void task3() {

    // Colors are defined as scalar
    Scalar clr(255, 255, 255);
    Scalar orange(0, 69, 255);
    Scalar black(0,0,0);

    // Defining an empty image, giving scalar to fill the image with the color
    // Type: CV_8UC3 8 bits U - unsingned, C3 - 3 channels (3 x (2^8=255))
    Mat img(512, 512, CV_8UC3, clr);

    // Thickness can be filled (enum FILLED = -1)
    circle(img, Point(256, 256), 155, orange, FILLED);
    // Rectangle
    rectangle(img, Point(130,226),Point(382, 286), clr, FILLED);
    // Line
    line(img, Point(130, 296), Point(382, 296), Scalar(255,255, 255), 2);
    // PutText put text to the image
    putText(img, "TASKS", Point(175,276),FONT_HERSHEY_PLAIN, 4, black, 5);
    imshow("Scalar", img);
    // imwrite(resourcePath + "blank.png", img);
    waitKey(0);
}

// Warping the image
void task4() {
    auto img = loadimg("cards.jpg");

    // Hardcoded width and height of output image
    auto w = 250.0f, h = 350.0f;

    // Src and DST warp points
    Point2f src[4] = {{529,142}, {771,190}, {405, 395}, {674, 457}};
    Point2f dst[4] = {{0.0f,0.0f}, {w,0.0f}, {0.0f, h}, {w, h}};

    Mat mtx = getPerspectiveTransform(src, dst);
    Mat imgWarp;

    // Warping from src to dst
    warpPerspective(img, imgWarp, mtx, Point(w,h));

    // Draw original warping points
    for (auto & i : src) {
        circle(img, i, 10, Scalar(0,0,255), FILLED);
    }

    imshow("Src", img);
    imshow("Warping", imgWarp);
    waitKey(0);

}

// Color masking
void task5() {
    Mat hsvImg, mask;
    auto img = loadimg("lambo.png");
    cvtColor(img, hsvImg, COLOR_BGR2HSV);

    // Define color in HSL min and max values
    int
            hmin = 0,
            smin = 110,
            vmin = 153,
            hmax = 19,
            smax = 240,
            vmax = 255;

    // Window with controls
    namedWindow("Trackbars", WINDOW_NORMAL);

    // Control trackbars (Runtime notice: passing values by address is not safe and depr, asks for callbacks)
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("Sat Min", "Trackbars", &smin, 255);
    createTrackbar("Sat Max", "Trackbars", &hmax, 255);
    createTrackbar("Val Min", "Trackbars", &vmin, 255);
    createTrackbar("Val Max", "Trackbars", &hmax, 255);
    // createTrackbar("Hue Min", "Trackbars", &hmin, 179);

    // Loop and act on images
    while(waitKey(1) != Q_KEY){
        Scalar lower(hmin, smin, vmin), upper(hmax, smax, vmax);
        imshow("Lambo", img);
        imshow("Lambo HSV", hsvImg);

        // Get a boolean image of the colors
        inRange(hsvImg, lower, upper, mask);
        imshow("Mask", mask);
    }
    destroyWindow("Lambo");
    destroyWindow("Lambo HSV");
    destroyWindow("Mask");
    destroyWindow("Trackbars");
}

void getContours(Mat& _img, Mat _draw) {


    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(_draw, contours, -1, Scalar(255,0,255), 2);
    vector<vector<Point>> conPoly;
    vector<Rect> boundRect;
    string objType;
    for (const vector<Point>& contour : contours)
    {
        vector<Point> _cont;
        auto area = contourArea(contour);
        // cout << area << endl;
        if(area > 1000.0) {
            auto peri = arcLength(contour, true);
            approxPolyDP(contour, _cont, 0.02*peri, true);

            //drawContours(_draw, vector<vector<Point>>(1, contour), -1, Scalar(255, 0, 255), 2);
            cout << "Polysize: " << _cont.size() << endl;
            conPoly.push_back(_cont);
            auto br = boundingRect(_cont);
            boundRect.push_back(br);
            int objCorner = (int)_cont.size();

            if (objCorner == 3) {
                // Triangle (3 corners)
                objType = "triangle";
            }
            if (objCorner == 4) {
                float aspRatio = (float)br.width / (float)br.height;
                cout << aspRatio << endl;
                // Square (4 corners)
                if (aspRatio > 0.9f && aspRatio < 1.1f) {
                    objType = "square";
                } else {
                    objType = "rect";
                }
            }
            if (objCorner > 4) {
                // More than 4
                objType = "circle";
            }
            putText(_draw, objType, Point(br.x, br.y - 5),FONT_HERSHEY_PLAIN, 1, Scalar(0,0,0), 1);

        }

    }
    // Draw bounding rectangles
    for (auto &rect : boundRect) {
        rectangle(_draw, rect, Scalar(0,0,255), 2);
    }

    // Draw contour polygons
//    drawContours(_draw, conPoly, -1, Scalar(255, 0, 255), 2);

    imshow("Contours", _draw);

}

void task6(){
    cout << "Running task 6: Shape Detection" << endl;

    auto img = loadimg("shapes.png");
    imshow("Shape detection", img);

    // Preprocess the image (find the shape by edge detection)
    Mat proc;
    cvtColor(img, proc, COLOR_BGR2GRAY);
    // imshow("Gray", proc);
    GaussianBlur(proc, proc, Size(3,3), 3, 0);
    // imshow("blurred", proc);
    Canny(proc, proc, 25, 75);
    // imshow("canny", proc);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // Fill gaps with dilation
    dilate(proc, proc, kernel);
    imshow("dilated", proc);
    Mat draw;
    getContours(proc, img);



    // erode(proc, proc, kernel);
    // imshow("eroded", proc);
    // imshow("processedImage", proc);

    waitKey(0);

}

void detectFaces(Mat img, CascadeClassifier &faceCascade){
    vector<Rect> faces;
    faceCascade.detectMultiScale(img, faces, 1.1f, 7);

    for (auto& face : faces) {
        rectangle(img, face.tl(), face.br(), Scalar(255, 0 ,255 ), 3);
    }
}

// Face detection
void task7(){
    auto img = loadimg("test.png");
    auto img2 = loadimg("ivan.jpg");
    resize(img2, img2, Size(), 0.5f, 0.5f);
    img2 = img2(Rect(200,50,500,400));
    auto img3 = loadimg("multiface.jpg");
    auto img4 = loadimg("chester.jpg");
//    resize(img3, img3, Size(), 0.5f, 0.5f);

    // Load model
    CascadeClassifier faceCascade;
    faceCascade.load(resourcePath + "haarcascade_frontalface_default.xml");
    if(faceCascade.empty()){
        cout << "FaceCascade could not be loaded!\n";
        throw bad_exception();
    }

    detectFaces(img, faceCascade);
    detectFaces(img2, faceCascade);
    detectFaces(img3, faceCascade);
    detectFaces(img4, faceCascade);

//    // Store bounding boxes
//    vector<Rect> faces;
//    faceCascade.detectMultiScale(img, faces, 1.1f, 10);
//
//    for (auto& face : faces) {
//        rectangle(img, face.tl(), face.br(), Scalar(255, 0 ,255 ), 3);
//    }

    imshow("FaceDetect", img);
    imshow("FaceDetect2", img2);
    imshow("FaceDetect3", img3);
    imshow("FaceDetect4", img4);
    waitKey(0);


}

Mat preProcess(Mat img){
    // Convert image color repr BGR -> GRAYSCALE
    Mat procImg;
    cvtColor(img, procImg, COLOR_BGR2GRAY);

    // Blur
    GaussianBlur(procImg, procImg, Size(127,127), 1, 1);

    // Edge detection
    Canny(procImg, procImg, 25, 75);

    // Fixing edge detection
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(procImg, procImg, kernel);
    // erode(img, img, kernel);
    return procImg;
}

vector<Point> getContours(Mat _img, bool draw, Mat _draw = Mat()) {


    vector<vector<Point>> contours;
    vector<Point> biggest;
    vector<Vec4i> hierarchy;

    findContours(_img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //drawContours(_draw, contours, -1, Scalar(255,0,255), 2);
    vector<vector<Point>> conPoly;
    vector<Rect> boundRect;
    string objType;
    double maxArea {0};
    for (const vector<Point>& contour : contours)
    {
        vector<Point> _cont;
        auto area = contourArea(contour);
        // cout << area << endl;
        if(area > 1000.0) {
            auto peri = arcLength(contour, true);
            approxPolyDP(contour, _cont, 0.02*peri, true);
            if (area > maxArea && _cont.size() == 4) {
                biggest = {_cont[0], _cont[1], _cont[2], _cont[3]};
                maxArea = area;
                if (draw && !_draw.empty()){
                    //drawContours(_draw, vector<vector<Point>>(1, contour), -1, Scalar(255, 0, 255), 2);
                }
            }
        }
            conPoly.push_back(_cont);
    }


    // Draw contour polygons
    // drawContours(_img, conPoly, -1, Scalar(255, 0, 255), 2);
    return biggest;
}

void drawPoints(Mat img, vector<Point> _points, Scalar clr){
    auto count = 0;
    for (auto& p : _points) {
        circle(img, p, 5, clr, FILLED);
        putText(img, to_string(count), p, FONT_HERSHEY_PLAIN, 2, clr, 2);
        ++count;
    }
}

vector<Point> reorderPoints(vector<Point> _p){
    vector<Point> res;
    map<int, Point> msum;
    map<int, Point> mdif;
    for (auto &p : _p){
        msum.insert(pair<int, Point>(p.x + p.y, p));
        mdif.insert(pair<int, Point>(p.x - p.y, p));
    }

    // first min(x+y)
    res.push_back( msum.begin()->second );
    // second max(x-y)
    res.push_back( mdif.rbegin()->second );
    // third min(x-y)
    res.push_back( mdif.begin()->second );
    // last max(x+y)
    res.push_back( msum.rbegin()->second );


    return res;

}

// Document scanner
void task8(){
    auto img = loadimg("paper.jpg");
    Mat imgGray, imgCanny; // Mattes


    // scaledown while working on image
    resize(img, img, Size(), 0.6f,0.5f);

    // Preprocessing (grayscale / blur / canny )
    auto imgDil = preProcess(img);
    auto initialPoints = getContours(imgDil, true, img);
    auto docPoints = reorderPoints(initialPoints);

    float w = 420, h = 596;
    auto imgWarp = getWarp(img, docPoints, w, h);

    Rect roi(5,5, w-(2*5), h-(2*5));
    auto imgCrop = imgWarp(roi);



    // Draw points to the original img
    drawPoints(img, initialPoints, Scalar(0,0,255));
    drawPoints(img, docPoints, Scalar(0,255,0));

    imshow("Document src", img);
    imshow("Document proc", imgDil);
    imshow("Warp", imgWarp);
    imshow("crop", imgCrop);

    waitKey(0);
}

Mat getWarp(Mat img, vector<Point> points, float w, float h) {
    Mat res;
    Point2f src[4] = {
            {static_cast<float>(points[0].x), static_cast<float>(points[0].y)},
            {static_cast<float>(points[1].x), static_cast<float>(points[1].y)},
            {static_cast<float>(points[2].x), static_cast<float>(points[2].y)},
            {static_cast<float>(points[3].x), static_cast<float>(points[3].y)}
    };
    Point2f dst[4] = {
            {0.0f, 0.0f},
            {w, 0.0f},
            {0.0f, h},
            {w, h}
    };

    auto matrix = getPerspectiveTransform(src, dst);
    warpPerspective(img, res, matrix, Point(w,h));

    return res;
}

void task9(){
    CascadeClassifier cascadeClassifier;
    cascadeClassifier.load(resourcePath + "haarcascade_russian_plate_number.xml");
    vector<Rect> plates;

    repeat:
        VideoCapture vcap(resourcePath + "plate_recognition.mp4");
        int fps = static_cast<int>(vcap.get(CAP_PROP_FPS));
        int vdelay = 1000/fps;
        cout << "Video framerate: " << to_string(fps) << "\n";
        Mat img;


            while(vcap.read(img)){
                cascadeClassifier.detectMultiScale(img, plates, 1.1, 10);

                for (auto& plate : plates){
                    // display plates
                    Mat plateImg = img(plate);
                    resize(plateImg, plateImg, Size(), 4.0f, 4.0f);
                    rectangle(img, plate.tl(), plate.br(), Scalar(255, 0, 255), 3);
                    imshow("Plate", plateImg);
                }


                imshow("Video", img);
                if (waitKey(20) == ESC_KEY) break;
            }
    if (waitKey(0) == 114){
        cout << "repeating video\n";
        goto repeat;
    } else {
        destroyWindow("Video");
        destroyWindow("Plate");
        return;
    }
}

void testvideo() {
    // Video


    VideoCapture vcap(resourcePath + "test_video.mp4");

    Mat img;

    while(vcap.read(img)){



        imshow("Image", img);
        waitKey(200);
    }
}

void testwebcam(){
    // Webcam
    VideoCapture webcap(0); // Use index of capture device
    Mat webimg;
    bool capture = false;
    while(webcap.read(webimg)){

        imshow("Image", webimg);
        waitKey(20);
    }
}

Mat loadimg(const string& rpath){
    return imread(resourcePath + rpath);
}

void testloadimg() {

    string imgPath = resourcePath + R"(test.png)";
    Mat test = imread(imgPath);
    imgPath = resourcePath + "cards.jpg";
    Mat cards = imread(imgPath);
    imgPath = resourcePath + "lambo.png";
    Mat lambo = imread(imgPath);
    imgPath = resourcePath + "paper.jpg";
    Mat paper = imread(imgPath);
    imgPath = resourcePath + "shapes.png";
    Mat shapes = imread(imgPath);
    imgPath = resourcePath + "signal.jpg";
    Mat signal = imread(imgPath);
    if(test.empty() && cards.empty() && lambo.empty() && paper.empty() && shapes.empty() && signal.empty()){
        throw bad_exception();
    }
}

int main() {
    // Images
    testloadimg();
    int key {0};
    imshow("Main",loadimg("blank.png"));
    while(key != ESC_KEY) {
         cout << key << endl;
        switch(key){
            case 49:
                task1();
                break;
            case 50:
                task2();
                break;
            case 51:
                task3();
                break;
            case 52:
                task4();
                break;
            case 53:
                task5();
                break;
            case 54:
                task6();
                break;
            case 55:
                task7();
                break;
            case 56:
                task8();
            case 57:
                task9();
            default:
                break;
        }
        key = waitKey(1001);
    }

    return 0;
}
