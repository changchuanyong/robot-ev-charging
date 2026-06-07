#include <Kinect.h>

#include <algorithm>
#include <cstdio>
#include <direct.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template<class Interface>
inline void SafeRelease(Interface*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

static void CreateDirectoryRecursive(const std::string& path) {
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        const char ch = path[i];
        current += ch;
        if (ch == '/' || ch == '\\') {
            _mkdir(current.c_str());
        }
    }
    _mkdir(current.c_str());
}

static std::string HrToHex(HRESULT hr) {
    std::ostringstream ss;
    ss << "0x" << std::hex << std::uppercase << static_cast<unsigned long>(hr);
    return ss.str();
}

static bool WriteStatus(
    const std::string& path,
    const std::string& status,
    bool sensorAvailable,
    int width,
    int height,
    int waitMs,
    int validCount,
    int totalCount,
    int minDepth,
    int maxDepth,
    double meanDepth,
    const std::string& message
) {
    std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
    if (!out) {
        return false;
    }

    out << std::fixed << std::setprecision(3);
    out << "{\n";
    out << "  \"status\": \"" << status << "\",\n";
    out << "  \"sensor_available\": " << (sensorAvailable ? "true" : "false") << ",\n";
    out << "  \"camera\": \"kinect_v2_depth\",\n";
    out << "  \"width\": " << width << ",\n";
    out << "  \"height\": " << height << ",\n";
    out << "  \"wait_ms\": " << waitMs << ",\n";
    out << "  \"valid_count\": " << validCount << ",\n";
    out << "  \"total_count\": " << totalCount << ",\n";
    out << "  \"valid_ratio\": " << (totalCount > 0 ? static_cast<double>(validCount) / totalCount : 0.0) << ",\n";
    out << "  \"min_depth_mm\": " << minDepth << ",\n";
    out << "  \"max_depth_mm\": " << maxDepth << ",\n";
    out << "  \"mean_depth_mm\": " << meanDepth << ",\n";
    out << "  \"message\": \"" << message << "\"\n";
    out << "}\n";
    return true;
}

int main() {
    const std::string outDir = "dataset\\live";
    const std::string statusPath = outDir + "\\latest_depth_status.json";
    CreateDirectoryRecursive(outDir);

    HRESULT hr = S_OK;
    IKinectSensor* sensor = nullptr;
    IDepthFrameSource* depthSource = nullptr;
    IDepthFrameReader* depthReader = nullptr;
    IFrameDescription* frameDescription = nullptr;

    int width = 0;
    int height = 0;
    BOOLEAN sensorAvailable = FALSE;

    hr = GetDefaultKinectSensor(&sensor);
    if (FAILED(hr) || sensor == nullptr) {
        const std::string msg = "GetDefaultKinectSensor failed: " + HrToHex(hr);
        WriteStatus(statusPath, "fail", false, width, height, 0, 0, 0, 0, 0, 0.0, msg);
        std::cout << msg << std::endl;
        return 1;
    }

    hr = sensor->Open();
    if (FAILED(hr)) {
        const std::string msg = "IKinectSensor::Open failed: " + HrToHex(hr);
        WriteStatus(statusPath, "fail", false, width, height, 0, 0, 0, 0, 0, 0.0, msg);
        SafeRelease(sensor);
        std::cout << msg << std::endl;
        return 1;
    }

    sensor->get_IsAvailable(&sensorAvailable);
    std::cout << "Kinect sensor available: " << (sensorAvailable ? "true" : "false") << std::endl;

    hr = sensor->get_DepthFrameSource(&depthSource);
    if (FAILED(hr) || depthSource == nullptr) {
        const std::string msg = "get_DepthFrameSource failed: " + HrToHex(hr);
        WriteStatus(statusPath, "fail", sensorAvailable == TRUE, width, height, 0, 0, 0, 0, 0, 0.0, msg);
        sensor->Close();
        SafeRelease(sensor);
        std::cout << msg << std::endl;
        return 1;
    }

    hr = depthSource->get_FrameDescription(&frameDescription);
    if (SUCCEEDED(hr) && frameDescription != nullptr) {
        frameDescription->get_Width(&width);
        frameDescription->get_Height(&height);
    }
    SafeRelease(frameDescription);

    hr = depthSource->OpenReader(&depthReader);
    if (FAILED(hr) || depthReader == nullptr) {
        const std::string msg = "Open depth reader failed: " + HrToHex(hr);
        WriteStatus(statusPath, "fail", sensorAvailable == TRUE, width, height, 0, 0, width * height, 0, 0, 0.0, msg);
        SafeRelease(depthSource);
        sensor->Close();
        SafeRelease(sensor);
        std::cout << msg << std::endl;
        return 1;
    }

    const int pollMs = 50;
    const int maxWaitMs = 5000;
    int waitedMs = 0;
    IDepthFrame* depthFrame = nullptr;

    while (waitedMs <= maxWaitMs) {
        hr = depthReader->AcquireLatestFrame(&depthFrame);
        if (SUCCEEDED(hr) && depthFrame != nullptr) {
            break;
        }
        SafeRelease(depthFrame);
        Sleep(pollMs);
        waitedMs += pollMs;
    }

    if (depthFrame == nullptr) {
        const std::string msg = "No depth frame received. Sensor may be disconnected or unavailable.";
        WriteStatus(statusPath, "no_frame", sensorAvailable == TRUE, width, height, waitedMs, 0, width * height, 0, 0, 0.0, msg);
        SafeRelease(depthReader);
        SafeRelease(depthSource);
        sensor->Close();
        SafeRelease(sensor);
        std::cout << msg << std::endl;
        std::cout << "Saved: " << statusPath << std::endl;
        return 2;
    }

    if (width <= 0 || height <= 0) {
        width = 512;
        height = 424;
    }

    std::vector<UINT16> depthData(static_cast<size_t>(width) * static_cast<size_t>(height));
    hr = depthFrame->CopyFrameDataToArray(static_cast<UINT>(depthData.size()), depthData.data());
    SafeRelease(depthFrame);

    if (FAILED(hr)) {
        const std::string msg = "CopyFrameDataToArray failed: " + HrToHex(hr);
        WriteStatus(statusPath, "fail", sensorAvailable == TRUE, width, height, waitedMs, 0, width * height, 0, 0, 0.0, msg);
        SafeRelease(depthReader);
        SafeRelease(depthSource);
        sensor->Close();
        SafeRelease(sensor);
        std::cout << msg << std::endl;
        return 1;
    }

    int validCount = 0;
    int minDepth = 0;
    int maxDepth = 0;
    double sumDepth = 0.0;

    for (UINT16 value : depthData) {
        if (value == 0) {
            continue;
        }
        const int depth = static_cast<int>(value);
        if (validCount == 0) {
            minDepth = depth;
            maxDepth = depth;
        } else {
            if (depth < minDepth) {
                minDepth = depth;
            }
            if (depth > maxDepth) {
                maxDepth = depth;
            }
        }
        sumDepth += depth;
        ++validCount;
    }

    const int totalCount = static_cast<int>(depthData.size());
    const double meanDepth = validCount > 0 ? sumDepth / validCount : 0.0;
    const std::string status = validCount > 0 ? "ok" : "empty";
    const std::string msg = validCount > 0 ? "Depth frame received." : "Depth frame received but contains no valid depth values.";

    WriteStatus(statusPath, status, sensorAvailable == TRUE, width, height, waitedMs, validCount, totalCount, minDepth, maxDepth, meanDepth, msg);

    std::cout << "Depth probe status: " << status << std::endl;
    std::cout << "size=" << width << "x" << height
              << " valid=" << validCount << "/" << totalCount
              << " min=" << minDepth << "mm"
              << " max=" << maxDepth << "mm"
              << " mean=" << std::fixed << std::setprecision(1) << meanDepth << "mm" << std::endl;
    std::cout << "Saved: " << statusPath << std::endl;

    SafeRelease(depthReader);
    SafeRelease(depthSource);
    sensor->Close();
    SafeRelease(sensor);
    return validCount > 0 ? 0 : 2;
}
