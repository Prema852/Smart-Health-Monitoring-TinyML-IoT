# Smart Health Monitoring and Emergency Alert System Using TinyML and IoT

## Project Overview
An ESP32-based smart health monitoring system that combines sensor data, TinyML-based human activity recognition and IoT connectivity. It monitors health parameters, recognizes physical activities and generates alerts when predefined abnormal conditions persist.

## Objectives
- Monitor pulse rate, ECG signals and body temperature using sensors.
- Recognize six human activities using a TinyML model deployed on ESP32.
- Implement activity-aware health monitoring with GPS-based location and GSM SMS alerts.
- Upload health and activity information to ThingSpeak for remote monitoring.

## Hardware Components
- ESP32-WROOM-32 – Main microcontroller
- Pulse Sensor – Pulse signal acquisition and BPM estimation
- AD8232 – ECG signal acquisition and lead-off detection
- DS18B20 – Body temperature measurement
- MPU6050 – Accelerometer and gyroscope data acquisition
- NEO-6M GPS – Location tracking
- SIM800C GSM – SMS notifications
- 16x2 I2C LCD – Local data display

## Software and Technologies
- Arduino IDE
- Embedded C/C++
- Python
- TensorFlow and Keras
- TensorFlow Lite and TensorFlow Lite Micro
- ThingSpeak IoT Platform
- NumPy and Matplotlib

## TinyML – Human Activity Recognition
- Dataset: UCI Human Activity Recognition Using Smartphones (UCI HAR).
- Model: 1D Convolutional Neural Network (CNN).
- Input: 128 samples × 6 channels.
- Sampling frequency: 50 Hz.
- Sensors: Three-axis accelerometer and gyroscope.
- Recognizes six activities:
  1. Walking
  2. Walking Upstairs
  3. Walking Downstairs
  4. Sitting
  5. Standing
  6. Laying
- Model converted using full integer INT8 quantization.
- Quantized model size: 42.9 KB.
- Model deployed locally on ESP32 using TensorFlow Lite Micro.

## Working Principle
1. Sensors collect pulse, ECG, temperature and motion data.
2. ESP32 processes the sensor readings.
3. MPU6050 data is divided into windows of 128 samples and passed to the TinyML model.
4. The CNN predicts the user's physical activity.
5. Activity-aware rule-based logic evaluates health readings against configured thresholds and persistence conditions.
6. If predefined abnormal conditions persist, the system generates an SMS alert with available GPS location information.
7. Health and activity data are uploaded to separate ThingSpeak channels.

## IoT Monitoring
Two ThingSpeak channels are used:

- **Health Monitoring Channel:** BPM, temperature, ECG readings, tilt angle, GPS coordinates, GPS status and health alert status.
- **Activity Recognition Channel:** Predicted activity, model score and completed inference windows.

## Project Features
- Multi-sensor health monitoring.
- On-device TinyML activity recognition.
- Activity-aware health alert logic.
- GPS-based location information.
- GSM-based SMS notifications.
- Real-time IoT data visualization through ThingSpeak.
- Local LCD display.


## Results
- Integrated multiple sensors with ESP32.
- Implemented six-class human activity recognition using a quantized 1D CNN.
- Verified model input and output dimensions.
- Implemented activity-aware health monitoring.
- Verified SMS notifications and ThingSpeak data uploads.

## Future Scope
- Improve activity recognition accuracy.
- Develop a mobile application for remote monitoring.
- Improve ECG signal processing and health analysis.
- Extend the system with validated fall detection.
- Perform extensive real-world testing and validation.

## Disclaimer
This project is an educational prototype and is not a certified medical device. Health thresholds are prototype settings, and the system is not clinically validated for diagnosis or emergency detection.
