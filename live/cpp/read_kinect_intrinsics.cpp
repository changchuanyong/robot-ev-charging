#include <Kinect.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

template<class Interface>
inline void SafeRelease(Interface*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

static void PrintHr(const char* label, HRESULT hr) {
    std::cout << label << " failed, HRESULT=0x"
              << std::hex << static_cast<unsigned long>(hr)
              << std::dec << std::endl;
}

int main() {
    const std::string outPath = "config\\kinect_v2_depth_intrinsics.json";

    HRESULT hr = S_OK;
    IKinectSensor* sensor = nullptr;
    ICoordinateMapper* mapper = nullptr;
    IDepthFrameSource* depthSource = nullptr;
    IDepthFrameReader* depthReader = nullptr;
    WAITABLE_HANDLE mappingChangedEvent = 0;

    hr = GetDefaultKinectSensor(&sensor);
    if (FAILED(hr) || sensor == nullptr) {
        PrintHr("GetDefaultKinectSensor", hr);
        return 1;
    }

    hr = sensor->Open();
    if (FAILED(hr)) {
        PrintHr("IKinectSensor::Open", hr);
        SafeRelease(sensor);
        return 1;
    }

    BOOLEAN isAvailable = FALSE;
    hr = sensor->get_IsAvailable(&isAvailable);
    if (SUCCEEDED(hr)) {
        std::cout << "Kinect sensor available: " << (isAvailable ? "true" : "false") << std::endl;
    }

    hr = sensor->get_CoordinateMapper(&mapper);
    if (FAILED(hr) || mapper == nullptr) {
        PrintHr("get_CoordinateMapper", hr);
        sensor->Close();
        SafeRelease(sensor);
        return 1;
    }

    hr = sensor->get_DepthFrameSource(&depthSource);
    if (FAILED(hr) || depthSource == nullptr) {
        PrintHr("get_DepthFrameSource", hr);
        SafeRelease(mapper);
        sensor->Close();
        SafeRelease(sensor);
        return 1;
    }

    hr = mapper->SubscribeCoordinateMappingChanged(&mappingChangedEvent);
    if (FAILED(hr)) {
        PrintHr("SubscribeCoordinateMappingChanged", hr);
    }

    hr = depthSource->OpenReader(&depthReader);
    if (FAILED(hr) || depthReader == nullptr) {
        PrintHr("Open depth frame reader", hr);
        if (mappingChangedEvent != 0) {
            mapper->UnsubscribeCoordinateMappingChanged(mappingChangedEvent);
        }
        SafeRelease(depthSource);
        SafeRelease(mapper);
        sensor->Close();
        SafeRelease(sensor);
        return 1;
    }

    std::cout << "Waiting for Kinect coordinate mapping calibration..." << std::endl;
    bool mappingReady = false;
    for (int i = 0; i < 600; ++i) {
        if (mappingChangedEvent != 0 &&
            WaitForSingleObject(reinterpret_cast<HANDLE>(mappingChangedEvent), 50) == WAIT_OBJECT_0) {
            ResetEvent(reinterpret_cast<HANDLE>(mappingChangedEvent));
            mappingReady = true;
            break;
        }

        IDepthFrame* depthFrame = nullptr;
        HRESULT frameHr = depthReader->AcquireLatestFrame(&depthFrame);
        if (SUCCEEDED(frameHr) && depthFrame != nullptr) {
            SafeRelease(depthFrame);
            mappingReady = true;
            break;
        }
        SafeRelease(depthFrame);
    }

    CameraIntrinsics intrinsics = {};
    hr = mapper->GetDepthCameraIntrinsics(&intrinsics);
    if (FAILED(hr)) {
        PrintHr("GetDepthCameraIntrinsics", hr);
        if (mappingChangedEvent != 0) {
            mapper->UnsubscribeCoordinateMappingChanged(mappingChangedEvent);
        }
        SafeRelease(depthReader);
        SafeRelease(depthSource);
        SafeRelease(mapper);
        sensor->Close();
        SafeRelease(sensor);
        return 1;
    }

    if (!mappingReady) {
        std::cout << "Warning: coordinate mapping event/frame was not observed before reading intrinsics." << std::endl;
    }

    if (intrinsics.FocalLengthX == 0.0f || intrinsics.FocalLengthY == 0.0f) {
        std::cout << "Invalid depth intrinsics returned by SDK: focal length is zero." << std::endl;
        std::cout << "Check that the Kinect service is running and the sensor is fully initialized." << std::endl;
        if (mappingChangedEvent != 0) {
            mapper->UnsubscribeCoordinateMappingChanged(mappingChangedEvent);
        }
        SafeRelease(depthReader);
        SafeRelease(depthSource);
        SafeRelease(mapper);
        sensor->Close();
        SafeRelease(sensor);
        return 2;
    }

    std::ofstream out(outPath.c_str(), std::ios::out | std::ios::trunc);
    if (!out) {
        std::cout << "Failed to open output file: " << outPath << std::endl;
        if (mappingChangedEvent != 0) {
            mapper->UnsubscribeCoordinateMappingChanged(mappingChangedEvent);
        }
        SafeRelease(depthReader);
        SafeRelease(depthSource);
        SafeRelease(mapper);
        sensor->Close();
        SafeRelease(sensor);
        return 1;
    }

    out << std::fixed << std::setprecision(8);
    out << "{\n";
    out << "  \"source\": \"Microsoft Kinect SDK v2 ICoordinateMapper::GetDepthCameraIntrinsics\",\n";
    out << "  \"camera\": \"kinect_v2_depth\",\n";
    out << "  \"image_width\": 512,\n";
    out << "  \"image_height\": 424,\n";
    out << "  \"fx\": " << intrinsics.FocalLengthX << ",\n";
    out << "  \"fy\": " << intrinsics.FocalLengthY << ",\n";
    out << "  \"cx\": " << intrinsics.PrincipalPointX << ",\n";
    out << "  \"cy\": " << intrinsics.PrincipalPointY << ",\n";
    out << "  \"radial_distortion_second_order\": " << intrinsics.RadialDistortionSecondOrder << ",\n";
    out << "  \"radial_distortion_fourth_order\": " << intrinsics.RadialDistortionFourthOrder << ",\n";
    out << "  \"radial_distortion_sixth_order\": " << intrinsics.RadialDistortionSixthOrder << "\n";
    out << "}\n";
    out.close();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Depth camera intrinsics read successfully." << std::endl;
    std::cout << "fx=" << intrinsics.FocalLengthX
              << " fy=" << intrinsics.FocalLengthY
              << " cx=" << intrinsics.PrincipalPointX
              << " cy=" << intrinsics.PrincipalPointY << std::endl;
    std::cout << "Saved: " << outPath << std::endl;

    if (mappingChangedEvent != 0) {
        mapper->UnsubscribeCoordinateMappingChanged(mappingChangedEvent);
    }
    SafeRelease(depthReader);
    SafeRelease(depthSource);
    SafeRelease(mapper);
    sensor->Close();
    SafeRelease(sensor);
    return 0;
}
