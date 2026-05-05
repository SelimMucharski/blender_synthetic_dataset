import depthai as dai
import numpy as np

# Connect to device and get calibration
with dai.Device() as device:
    calib_data = device.readCalibration()
    
    # Get Intrinsics/Distortion for a socket (e.g., CAM_A)
    intrinsics = np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A))
    dist_coeffs = np.array(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
