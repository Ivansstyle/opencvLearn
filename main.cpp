#include <iostream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


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

Mat loadimg(const string&);

// image manipulation
void task1() {
    auto img = loadimg("test.png");
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgGray, Size(127,127), 1, 1);
    Canny(imgGray, imgGray, 25, 75);

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
    resize(img, imgResize, Size(), .5, .5);
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
    // Type: CV_8UC3 8 bits U - unsingned, C3 - 3 channels (3 x (2^8=255))
    Mat img(512, 512, CV_8UC3, clr);

    // Thickness can be filled (enum FILLED = -1)
    circle(img, Point(256, 256), 155, orange, FILLED);
    // Rectangle
    rectangle(img, Point(130,226),Point(382, 286), clr, FILLED);
    // Line
    line(img, Point(130, 296), Point(382, 296), Scalar(255,255, 255), 2);
    // PutText
    putText(img, "TASKS", Point(175,276),FONT_HERSHEY_PLAIN, 4, black, 5);
    imshow("Scalar", img);
    // imwrite(resourcePath + "blank.png", img);
    waitKey(0);
}

// Warping the image
void task4() {
    auto img = loadimg("cards.jpg");

    auto w = 250.0f, h = 350.0f;

    Point2f src[4] = {{529,142}, {771,190}, {405, 395}, {674, 457}};
    Point2f dst[4] = {{0.0f,0.0f}, {w,0.0f}, {0.0f, h}, {w, h}};

    Mat mtx = getPerspectiveTransform(src, dst);
    Mat imgWarp;
    warpPerspective(img, imgWarp, mtx, Point(w,h));

    for (auto & i : src) {
        circle(img, i, 10, Scalar(0,0,255), FILLED);
    }

    imshow("Src", img);

    imshow("Warping", imgWarp);
    waitKey(0);

}
// Color detection
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

    namedWindow("Trackbars", WINDOW_NORMAL);
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("Sat Min", "Trackbars", &smin, 255);
    createTrackbar("Sat Max", "Trackbars", &hmax, 255);
    createTrackbar("Val Min", "Trackbars", &vmin, 255);
    createTrackbar("Val Max", "Trackbars", &hmax, 255);
    // createTrackbar("Hue Min", "Trackbars", &hmin, 179);

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

void testvideo() {
    // Video
    VideoCapture vcap(resourcePath + "test_video.mp4");
    Mat vimg;
    while(vcap.read(vimg)){
        imshow("Image", vimg);
        waitKey(20);
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
        // cout << key << endl;
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
            default:
                break;
        }
        key = waitKey(1001);
    }

    return 0;
}
