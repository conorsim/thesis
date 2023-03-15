#include <iostream>

// include depthai library
// #include "depthai/depthai.hpp"
#include <depthai/depthai.hpp>

// include opencv library (Optional, used only for the following example)
#include <opencv2/opencv.hpp>

#include "deque"
#include "unordered_map"
#include "unordered_set"

static const auto lineColor = cv::Scalar(200, 0, 200);
static const auto pointColor = cv::Scalar(0, 0, 255);

class FeatureTrackerDrawer {
   private:
    static const int circleRadius = 2;
    static const int maxTrackedFeaturesPathLength = 30;
    // for how many frames the feature is tracked
    static int trackedFeaturesPathLength;

    using featureIdType = decltype(dai::Point2f::x);

    std::unordered_set<featureIdType> trackedIDs;
    std::unordered_map<featureIdType, std::deque<dai::Point2f>> trackedFeaturesPath;

    std::string trackbarName;
    std::string windowName;

   public:
    void trackFeaturePath(std::vector<dai::TrackedFeature>& features) {
        std::unordered_set<featureIdType> newTrackedIDs;
        for(auto& currentFeature : features) {
            auto currentID = currentFeature.id;
            newTrackedIDs.insert(currentID);

            if(!trackedFeaturesPath.count(currentID)) {
                trackedFeaturesPath.insert({currentID, std::deque<dai::Point2f>()});
            }
            std::deque<dai::Point2f>& path = trackedFeaturesPath.at(currentID);

            path.push_back(currentFeature.position);
            while(path.size() > std::max<unsigned int>(1, trackedFeaturesPathLength)) {
                path.pop_front();
            }
        }

        std::unordered_set<featureIdType> featuresToRemove;
        for(auto& oldId : trackedIDs) {
            if(!newTrackedIDs.count(oldId)) {
                featuresToRemove.insert(oldId);
            }
        }

        for(auto& id : featuresToRemove) {
            trackedFeaturesPath.erase(id);
        }

        trackedIDs = newTrackedIDs;
    }

    void drawFeatures(cv::Mat& img) {
        cv::setTrackbarPos(trackbarName.c_str(), windowName.c_str(), trackedFeaturesPathLength);

        for(auto& featurePath : trackedFeaturesPath) {
            std::deque<dai::Point2f>& path = featurePath.second;
            unsigned int j = 0;
            for(j = 0; j < path.size() - 1; j++) {
                auto src = cv::Point(path[j].x, path[j].y);
                auto dst = cv::Point(path[j + 1].x, path[j + 1].y);
                cv::line(img, src, dst, lineColor, 1, cv::LINE_AA, 0);
            }

            cv::circle(img, cv::Point(path[j].x, path[j].y), circleRadius, pointColor, -1, cv::LINE_AA, 0);
        }
    }

    FeatureTrackerDrawer(std::string trackbarName, std::string windowName) {
        this->trackbarName = trackbarName;
        this->windowName = windowName;
        cv::namedWindow(windowName.c_str());
        cv::createTrackbar(trackbarName.c_str(), windowName.c_str(), &trackedFeaturesPathLength, maxTrackedFeaturesPathLength, nullptr);
    }
};

int FeatureTrackerDrawer::trackedFeaturesPathLength = 10;

int main(){
    using namespace std;

    // Create pipeline
    dai::Pipeline pipeline;
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    // auto neuralNetwork = pipeline.create<dai::node::NeuralNetwork>();
    auto featureTrackerRight = pipeline.create<dai::node::FeatureTracker>();

    auto xoutRight = pipeline.create<dai::node::XLinkOut>();
    auto xoutDisp = pipeline.create<dai::node::XLinkOut>();
    auto xoutNN = pipeline.create<dai::node::XLinkOut>();
    auto xoutTracker = pipeline.create<dai::node::XLinkOut>();

    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);

    stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DENSITY);
    stereo->setLeftRightCheck(true);
    stereo->setExtendedDisparity(true);
    // stereo->setSubpixel(true)

    // neuralNetwork->setBlobPath("model.blob");

    xoutRight->setStreamName("right");
    xoutDisp->setStreamName("disp");
    xoutNN->setStreamName("nn");
    xoutTracker->setStreamName("trackedFeaturesRight");

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->rectifiedRight.link(xoutRight->input);
    stereo->disparity.link(xoutDisp->input);
    // neuralNetwork->out.link(xoutNN->input);

    // stereo->rectifiedRight.link(neuralNetwork->inputs["right"]);
    // stereo->disparity.link(neuralNetwork->inputs["disp"]);

    stereo->rectifiedRight.link(featureTrackerRight->inputImage);
    featureTrackerRight->outputFeatures.link(xoutTracker->input);

    // By default the least mount of resources are allocated
    // increasing it improves performance when optical flow is enabled
    auto numShaves = 2;
    auto numMemorySlices = 2;
    featureTrackerRight->setHardwareResources(numShaves, numMemorySlices);

    const auto rightWindowName = "right";
    auto rightFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", rightWindowName);

    try {
        // Try connecting to device and start the pipeline
        dai::Device device(pipeline);

        // Get output queue
        auto qRight = device.getOutputQueue("right");
        auto qDisp = device.getOutputQueue("disp");
        // auto qNN = device.getOutputQueue("nn");
        auto qFeatures = device.getOutputQueue("trackedFeaturesRight");

        cv::Mat right;
        cv::Mat disp;
        cv::Mat dispColor;
        while (true) {

            // Receive 'preview' frame from device
            auto imgFrame = qRight->get<dai::ImgFrame>();
            auto dispFrame = qDisp->get<dai::ImgFrame>();
            // auto inNN = qNN->get<dai::NNData>();
            auto trackedFeaturesRight = qFeatures->get<dai::TrackedFeatures>()->trackedFeatures;

            rightFeatureDrawer.trackFeaturePath(trackedFeaturesRight);
            rightFeatureDrawer.drawFeatures(right);

            // vector<string> layer_names = inNN->getAllLayerNames();
            // vector<float> test = inNN->getLayerFp16(layer_names[0]);

            // std::cout << typeid(test).name() << endl;
            // std::cout << test.size() << endl;

            right = imgFrame->getCvFrame();
            disp = dispFrame->getCvFrame();
            disp.convertTo(disp, CV_8UC1, 255 / stereo->initialConfig.getMaxDisparity());

            cv::applyColorMap(disp, dispColor, cv::COLORMAP_JET);

            // Show the received 'preview' frame
            cv::imshow("right", right);
            cv::imshow("disparity", dispColor);

            // Wait and check if 'q' pressed
            if (cv::waitKey(1) == 'q') return 0;

        }
    } catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
    }


    return 0;
}
