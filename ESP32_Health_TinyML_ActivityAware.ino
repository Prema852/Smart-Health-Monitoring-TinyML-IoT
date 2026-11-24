/*
  ESP32 HEALTH MONITORING + TINYML ACTIVITY RECOGNITION
  Activity-aware health alert prototype.

  Keep activity_model.h in this sketch folder.

  HEALTH CHANNEL:
  Field 1: Pulse BPM
  Field 2: Temperature (invalid sensor reading sent as 0)
  Field 3: ECG raw
  Field 4: Tilt angle
  Field 5: Latitude
  Field 6: Longitude
  Field 7: GPS status
  Field 8: Health alert (0/1)

  ACTIVITY CHANNEL:
  Field 1: Activity index (0 to 5; -1 until first prediction)
  Field 2: Model output score
  Field 3: Completed windows

  IMPORTANT:
  - Thresholds below are prototype settings, NOT medical diagnosis limits.
  - Validate the pulse sensor and body-temperature measurement method.
  - The UCI-HAR model has limited sitting/standing/laying accuracy.
  - Wi-Fi/ThingSpeak operations can still interrupt 50 Hz sampling.
*/

#include <WiFi.h>
#include <ThingSpeak.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <TinyGPSPlus.h>
#include <math.h>

#include <Chirale_TensorFlowLite.h>
#include "activity_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// =====================================================
// CREDENTIALS: FILL THESE IN PRIVATELY
// =====================================================

const char* ssid = "Prema";
const char* password = "premahmk";

unsigned long healthChannelID = 3595422;
const char* healthWriteKey = "01JQA19VQU4KABXP";

unsigned long activityChannelID = 3595540;
const char* activityWriteKey = "TU5ARVBHKAF0DMUN";

const char* phoneNumber = "7760051573";

WiFiClient client;

// =====================================================
// LCD AND PINS
// =====================================================

LiquidCrystal_I2C lcd(0x27, 16, 2);

#define PULSE_PIN 34
#define ECG_PIN 35
#define ECG_LO_PLUS 32
#define ECG_LO_MINUS 33
#define TEMP_PIN 4

#define GPS_RX 26
#define GPS_TX 27
#define GSM_RX 16
#define GSM_TX 17

#define MPU_ADDR 0x68

// =====================================================
// SENSOR OBJECTS
// =====================================================

OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);
TinyGPSPlus gps;

HardwareSerial GPSserial(1);
HardwareSerial GSMserial(2);

// =====================================================
// HEALTH MONITORING VARIABLES
// =====================================================

float temperature = -127.0;
float pulseBPM = 0;
float tiltAngle = 0;
int ecgValue = 0;

float latitude = 0;
float longitude = 0;

bool gpsAvailable = false;
bool ecgLeadOff = true;
bool mpuAvailable = false;
bool healthAlert = false;

unsigned long lastBeatTime = 0;
unsigned long lastValidBeatTime = 0;
unsigned long lastPulseDebug = 0;
unsigned long lastECGDebug = 0;
unsigned long lastThingSpeak = 0;
unsigned long lastSMS = 0;

const int PULSE_THRESHOLD = 2000;
const int PULSE_HYSTERESIS = 100;
bool pulseAboveThreshold = false;

const unsigned long SMS_INTERVAL = 300000UL;

bool temperatureConversionPending = false;
unsigned long temperatureRequestTime = 0;
const unsigned long TEMP_CONVERSION_TIME = 750UL;

// =====================================================
// ACTIVITY-AWARE ALERT SETTINGS
// These are configurable prototype values, not medical
// diagnosis thresholds.
// =====================================================

const float RESTING_BPM_LIMIT = 100.0;
const float ACTIVE_BPM_LIMIT = 140.0;
const float LOW_BPM_LIMIT = 45.0;
const float HIGH_TEMP_LIMIT = 38.0;

const unsigned long ALERT_PERSISTENCE = 30000UL;

unsigned long highBpmStart = 0;
unsigned long lowBpmStart = 0;
unsigned long highTempStart = 0;

int previousHealthActivity = -1;

// =====================================================
// NON-BLOCKING SMS STATE
// =====================================================

bool smsPending = false;
String pendingSMSMessage = "";

enum SmsState {
  SMS_IDLE,
  SMS_WAIT_AT,
  SMS_WAIT_TEXT_MODE,
  SMS_WAIT_PROMPT,
  SMS_WAIT_SEND
};

SmsState smsState = SMS_IDLE;
unsigned long smsStateStarted = 0;

// =====================================================
// TINYML CONFIGURATION
// =====================================================

#define WINDOW_SIZE 128
#define SENSOR_CHANNELS 6

const unsigned long SAMPLE_INTERVAL_US = 20000UL;

const float ALPHA = 0.9630;
const float RAD_CONVERSION = 0.01745329252;

const char* activities[6] = {
  "WALKING",
  "UPSTAIRS",
  "DOWNSTAIRS",
  "SITTING",
  "STANDING",
  "LAYING"
};

float gravity[3] = {0, 0, 0};
float activitySamples[WINDOW_SIZE][SENSOR_CHANNELS];

int activitySampleCount = 0;
unsigned long completedWindows = 0;
unsigned long lastMpuSampleUs = 0;

bool gravityInitialized = false;

int predictedIndex = -1;
float predictedConfidence = 0.0;
const char* predictedActivity = "COLLECTING";

// Consecutive-window activity confirmation.
// Lower this only after checking model output scores.
// Scores are quantized model outputs, not clinical probabilities.
const float MIN_ACTIVITY_SCORE = 0.30;
const int REQUIRED_ACTIVITY_WINDOWS = 3;

int candidateActivityIndex = -1;
int stableActivityIndex = -1;
int consecutiveActivityWindows = 0;

// =====================================================
// TFLITE MODEL
// =====================================================

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// =====================================================
// MPU REGISTER WRITE
// =====================================================

bool writeMPURegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

// =====================================================
// MPU6050 INITIALIZATION
// =====================================================

void initMPU6050() {
  if (!writeMPURegister(0x6B, 0x01)) {
    mpuAvailable = false;
    Serial.println("MPU initialization failed");
    return;
  }

  delay(100);

  writeMPURegister(0x1A, 0x03);  // Digital low-pass filter
  writeMPURegister(0x19, 19);    // 50 Hz sample rate
  writeMPURegister(0x1B, 0x00);  // Gyro +/-250 dps
  writeMPURegister(0x1C, 0x00);  // Accel +/-2g
  writeMPURegister(0x6C, 0x00);

  delay(100);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);

  if (Wire.endTransmission(false) != 0) {
    mpuAvailable = false;
    Serial.println("MPU6050 not responding");
    return;
  }

  if (Wire.requestFrom(MPU_ADDR, 1, true) != 1) {
    mpuAvailable = false;
    Serial.println("MPU identity read failed");
    return;
  }

  uint8_t deviceID = Wire.read();

  if (deviceID == 0x68) {
    mpuAvailable = true;
    Serial.println("MPU6050 initialized");
  } else {
    mpuAvailable = false;
    Serial.print("Unexpected MPU ID: ");
    Serial.println(deviceID, HEX);
  }
}

// =====================================================
// READ MPU6050 SIX-AXIS DATA
// =====================================================

bool readMPURegisters(
  float &ax, float &ay, float &az,
  float &gx, float &gy, float &gz
) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);

  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, 14, true) != 14) return false;

  int16_t rawAx = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawAy = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawAz = (int16_t)((Wire.read() << 8) | Wire.read());

  Wire.read();
  Wire.read();

  int16_t rawGx = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawGy = (int16_t)((Wire.read() << 8) | Wire.read());
  int16_t rawGz = (int16_t)((Wire.read() << 8) | Wire.read());

  ax = rawAx / 16384.0;
  ay = rawAy / 16384.0;
  az = rawAz / 16384.0;

  gx = (rawGx / 131.0) * RAD_CONVERSION;
  gy = (rawGy / 131.0) * RAD_CONVERSION;
  gz = (rawGz / 131.0) * RAD_CONVERSION;

  return true;
}

// =====================================================
// UPDATE STABLE ACTIVITY
// =====================================================

void updateStableActivity() {
  if (predictedIndex < 0 ||
      predictedIndex >= 6 ||
      predictedConfidence < MIN_ACTIVITY_SCORE) {

    candidateActivityIndex = -1;
    consecutiveActivityWindows = 0;
    stableActivityIndex = -1;
    return;
  }

  if (predictedIndex == candidateActivityIndex) {
    consecutiveActivityWindows++;
  } else {
    candidateActivityIndex = predictedIndex;
    consecutiveActivityWindows = 1;
  }

  if (consecutiveActivityWindows >= REQUIRED_ACTIVITY_WINDOWS) {
    stableActivityIndex = predictedIndex;
  }
}

// =====================================================
// TINYML PREDICTION
// =====================================================

void runPrediction() {
  Serial.println("\n128 samples collected!");
  Serial.println("Running activity prediction...");

  for (int i = 0; i < WINDOW_SIZE; i++) {
    for (int j = 0; j < SENSOR_CHANNELS; j++) {
      float value = activitySamples[i][j];

      int quantized = (int)roundf(
        value / input->params.scale
      ) + input->params.zero_point;

      if (quantized > 127) quantized = 127;
      if (quantized < -128) quantized = -128;

      input->data.int8[i * SENSOR_CHANNELS + j] =
        (int8_t)quantized;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Inference failed!");

    predictedIndex = -1;
    predictedActivity = "ERROR";
    predictedConfidence = 0;

    candidateActivityIndex = -1;
    consecutiveActivityWindows = 0;
    stableActivityIndex = -1;
    return;
  }

  predictedIndex = 0;
  predictedConfidence = -1000;

  Serial.println("Activity scores:");

  for (int i = 0; i < 6; i++) {
    int8_t raw = output->data.int8[i];

    float score =
      (raw - output->params.zero_point) *
      output->params.scale;

    Serial.print(activities[i]);
    Serial.print(": ");
    Serial.println(score, 4);

    if (score > predictedConfidence) {
      predictedConfidence = score;
      predictedIndex = i;
    }
  }

  predictedActivity = activities[predictedIndex];

  updateStableActivity();

  Serial.print("PREDICTED ACTIVITY: ");
  Serial.println(predictedActivity);

  Serial.print("Output score: ");
  Serial.println(predictedConfidence, 4);

  Serial.print("Stable activity: ");
  if (stableActivityIndex >= 0) {
    Serial.println(activities[stableActivityIndex]);
  } else {
    Serial.println("NOT CONFIRMED");
  }

  Serial.println("-------------------------");

  completedWindows++;
}

// =====================================================
// MPU6050 SAMPLING AND PREPROCESSING
// =====================================================

void readMPU6050() {
  if (!mpuAvailable || !gravityInitialized) return;

  unsigned long nowUs = micros();

  if (nowUs - lastMpuSampleUs < SAMPLE_INTERVAL_US) return;

  lastMpuSampleUs = nowUs;

  float ax, ay, az;
  float gx, gy, gz;

  if (!readMPURegisters(ax, ay, az, gx, gy, gz)) {
    Serial.println("MPU read failed. Resetting window.");

    mpuAvailable = false;
    activitySampleCount = 0;
    return;
  }

  tiltAngle = atan2(
    ax,
    sqrt(ay * ay + az * az)
  ) * 180.0 / PI;

  gravity[0] = ALPHA * gravity[0] + (1.0 - ALPHA) * ax;
  gravity[1] = ALPHA * gravity[1] + (1.0 - ALPHA) * ay;
  gravity[2] = ALPHA * gravity[2] + (1.0 - ALPHA) * az;

  float bodyAx = ax - gravity[0];
  float bodyAy = ay - gravity[1];
  float bodyAz = az - gravity[2];

  activitySamples[activitySampleCount][0] = bodyAx;
  activitySamples[activitySampleCount][1] = bodyAy;
  activitySamples[activitySampleCount][2] = bodyAz;
  activitySamples[activitySampleCount][3] = gx;
  activitySamples[activitySampleCount][4] = gy;
  activitySamples[activitySampleCount][5] = gz;

  activitySampleCount++;

  if (activitySampleCount >= WINDOW_SIZE) {
    runPrediction();
    activitySampleCount = 0;
  }
}

// =====================================================
// PULSE SENSOR
// =====================================================

void readPulse() {
  int pulseValue = analogRead(PULSE_PIN);

  if (millis() - lastPulseDebug >= 500) {
    Serial.print("Pulse sensor raw value: ");
    Serial.println(pulseValue);
    lastPulseDebug = millis();
  }

  if (pulseValue > PULSE_THRESHOLD) {
    if (!pulseAboveThreshold) {
      pulseAboveThreshold = true;

      unsigned long now = millis();

      if (lastBeatTime > 0) {
        unsigned long interval = now - lastBeatTime;

        if (interval >= 300 && interval <= 2000) {
          pulseBPM = 60000.0 / interval;
          lastValidBeatTime = now;
        }
      }

      lastBeatTime = now;
    }
  }

  if (pulseValue < PULSE_THRESHOLD - PULSE_HYSTERESIS) {
    pulseAboveThreshold = false;
  }

  if (lastValidBeatTime == 0 ||
      millis() - lastValidBeatTime > 3000) {
    pulseBPM = 0;
  }
}

// =====================================================
// ECG SENSOR
// =====================================================

void readECG() {
  int loPlus = digitalRead(ECG_LO_PLUS);
  int loMinus = digitalRead(ECG_LO_MINUS);

  ecgLeadOff = (loPlus == HIGH || loMinus == HIGH);
  ecgValue = analogRead(ECG_PIN);

  if (millis() - lastECGDebug >= 500) {
    if (ecgLeadOff) {
      Serial.println("ECG: LEADS DISCONNECTED");
    } else {
      Serial.print("ECG Raw Value: ");
      Serial.println(ecgValue);
    }

    lastECGDebug = millis();
  }
}

// =====================================================
// TEMPERATURE SENSOR - NON-BLOCKING CONVERSION
// =====================================================

void readTemperature() {
  if (!temperatureConversionPending) {
    sensors.requestTemperatures();
    temperatureRequestTime = millis();
    temperatureConversionPending = true;
    return;
  }

  if (millis() - temperatureRequestTime >= TEMP_CONVERSION_TIME) {
    float newTemperature = sensors.getTempCByIndex(0);

    if (newTemperature == DEVICE_DISCONNECTED_C ||
        newTemperature < -55 ||
        newTemperature > 125 ||
        fabs(newTemperature - 85.0) < 0.01) {
      temperature = -127.0;
    } else {
      temperature = newTemperature;
    }

    temperatureConversionPending = false;
  }
}

// =====================================================
// GPS
// =====================================================

void readGPS() {
  while (GPSserial.available()) {
    gps.encode(GPSserial.read());
  }

  if (gps.location.isValid() && gps.location.age() < 5000) {
    latitude = gps.location.lat();
    longitude = gps.location.lng();
    gpsAvailable = true;
  } else {
    gpsAvailable = false;
  }
}

// =====================================================
// ACTIVITY-AWARE HEALTH ANALYSIS
// =====================================================

void checkHealth() {
  healthAlert = false;

  String alertMessage = "HEALTH ALERT: ";
  unsigned long now = millis();

  bool activityConfirmed = (stableActivityIndex >= 0);

  bool resting =
    activityConfirmed &&
    (stableActivityIndex == 3 || stableActivityIndex == 5);

  bool physicallyActive =
    activityConfirmed &&
    (stableActivityIndex == 0 ||
     stableActivityIndex == 1 ||
     stableActivityIndex == 2);

  // Reset BPM persistence when the confirmed activity changes.
  if (stableActivityIndex != previousHealthActivity) {
    highBpmStart = 0;
    previousHealthActivity = stableActivityIndex;
  }

  // Pulse sensor must have a recent, plausible beat reading.
  bool validBPM =
    pulseBPM >= 35 &&
    pulseBPM <= 220 &&
    lastValidBeatTime > 0 &&
    now - lastValidBeatTime <= 3000UL;

  bool highBPM = false;

  if (validBPM && resting &&
      pulseBPM > RESTING_BPM_LIMIT) {
    highBPM = true;
  }

  if (validBPM && physicallyActive &&
      pulseBPM > ACTIVE_BPM_LIMIT) {
    highBPM = true;
  }

  if (highBPM) {
    if (highBpmStart == 0) highBpmStart = now;

    if (now - highBpmStart >= ALERT_PERSISTENCE) {
      healthAlert = true;
      alertMessage += "Elevated BPM during confirmed activity. ";
    }
  } else {
    highBpmStart = 0;
  }

  // Low BPM check is independent of activity.
  if (validBPM && pulseBPM < LOW_BPM_LIMIT) {
    if (lowBpmStart == 0) lowBpmStart = now;

    if (now - lowBpmStart >= ALERT_PERSISTENCE) {
      healthAlert = true;
      alertMessage += "Low BPM for sustained period. ";
    }
  } else {
    lowBpmStart = 0;
  }

  // Temperature check is independent of activity.
  bool validTemperature =
    temperature > -55 &&
    temperature < 125 &&
    temperature != -127.0 &&
    fabs(temperature - 85.0) > 0.01;

  if (validTemperature && temperature >= HIGH_TEMP_LIMIT) {
    if (highTempStart == 0) highTempStart = now;

    if (now - highTempStart >= ALERT_PERSISTENCE) {
      healthAlert = true;
      alertMessage += "Elevated temperature for sustained period. ";
    }
  } else {
    highTempStart = 0;
  }

  // Tilt is displayed, but is not treated as a medical diagnosis
  // or fall detection.
  if (gpsAvailable && healthAlert) {
    alertMessage += "Location: https://maps.google.com/?q=";
    alertMessage += String(latitude, 6);
    alertMessage += ",";
    alertMessage += String(longitude, 6);
  }

  // Queue one SMS; cooldown prevents repeated messages.
  if (healthAlert &&
      !smsPending &&
      smsState == SMS_IDLE &&
      (lastSMS == 0 || now - lastSMS >= SMS_INTERVAL)) {

    smsPending = true;
    pendingSMSMessage = alertMessage;

    Serial.println("HEALTH ALERT DETECTED!");
    Serial.println(alertMessage);
  }
}

// =====================================================
// NON-BLOCKING GSM SMS SERVICE
// =====================================================

void servicePendingSMS() {
  unsigned long now = millis();

  if (smsState == SMS_IDLE) {
    if (!smsPending) return;

    if (lastSMS != 0 && now - lastSMS < SMS_INTERVAL) return;

    GSMserial.println("AT");
    smsState = SMS_WAIT_AT;
    smsStateStarted = now;
    return;
  }

  if (smsState == SMS_WAIT_AT &&
      now - smsStateStarted >= 500) {
    GSMserial.println("AT+CMGF=1");
    smsState = SMS_WAIT_TEXT_MODE;
    smsStateStarted = now;
    return;
  }

  if (smsState == SMS_WAIT_TEXT_MODE &&
      now - smsStateStarted >= 500) {
    GSMserial.print("AT+CMGS=\"");
    GSMserial.print(phoneNumber);
    GSMserial.println("\"");

    smsState = SMS_WAIT_PROMPT;
    smsStateStarted = now;
    return;
  }

  if (smsState == SMS_WAIT_PROMPT &&
      now - smsStateStarted >= 1200) {
    GSMserial.print(pendingSMSMessage);
    GSMserial.write(26);

    smsState = SMS_WAIT_SEND;
    smsStateStarted = now;
    return;
  }

  if (smsState == SMS_WAIT_SEND &&
      now - smsStateStarted >= 5000) {
    Serial.println("SMS send sequence completed (modem response not verified).");

    lastSMS = now;
    smsPending = false;
    pendingSMSMessage = "";
    smsState = SMS_IDLE;
  }
}

// =====================================================
// LCD DISPLAY
// =====================================================

void displayLCD() {
  static unsigned long lastPage = 0;
  static int page = 0;

  if (millis() - lastPage >= 3000) {
    lastPage = millis();
    page = (page + 1) % 4;
    lcd.clear();
  }

  if (page == 0) {
    lcd.setCursor(0, 0);
    lcd.print("Pulse: ");
    lcd.print(pulseBPM, 0);
    lcd.print(" BPM   ");

    lcd.setCursor(0, 1);
    lcd.print("Temp: ");

    if (temperature == -127.0) {
      lcd.print("ERROR     ");
    } else {
      lcd.print(temperature, 1);
      lcd.print(" C       ");
    }
  }
  else if (page == 1) {
    lcd.setCursor(0, 0);
    lcd.print("ECG: ");
    lcd.print(ecgValue);
    lcd.print("     ");

    lcd.setCursor(0, 1);
    lcd.print(ecgLeadOff ? "ECG LEADS OFF " : "ECG CONNECTED ");
  }
  else if (page == 2) {
    lcd.setCursor(0, 0);
    lcd.print("Activity:");

    lcd.setCursor(0, 1);

    if (stableActivityIndex >= 0) {
      lcd.print(activities[stableActivityIndex]);
    } else {
      lcd.print("Checking...");
    }
    lcd.print("                ");
  }
  else if (page == 3) {
    lcd.setCursor(0, 0);
    lcd.print(healthAlert ? "HEALTH ALERT" : "Status: Monitor");

    lcd.setCursor(0, 1);
    lcd.print("GPS:");
    lcd.print(gpsAvailable ? "OK " : "NO ");
    lcd.print("Tilt:");
    if (mpuAvailable) {
      lcd.print(tiltAngle, 0);
      lcd.print("   ");
    } else {
      lcd.print("ERR ");
    }
  }
}

// =====================================================
// HEALTH THINGSPEAK CHANNEL
// =====================================================

void updateHealthThingSpeak() {
  if (WiFi.status() != WL_CONNECTED) return;

  ThingSpeak.setField(1, pulseBPM);
  ThingSpeak.setField(2, (temperature == -127.0) ? 0.0f : temperature);
  ThingSpeak.setField(3, ecgValue);
  ThingSpeak.setField(4, tiltAngle);
  ThingSpeak.setField(5, latitude);
  ThingSpeak.setField(6, longitude);
  ThingSpeak.setField(7, gpsAvailable ? 1 : 0);
  ThingSpeak.setField(8, healthAlert ? 1 : 0);

  int result = ThingSpeak.writeFields(
    healthChannelID,
    healthWriteKey
  );

  Serial.print("Health ThingSpeak response: ");
  Serial.println(result);
}

// =====================================================
// ACTIVITY THINGSPEAK CHANNEL
// =====================================================

void updateActivityThingSpeak() {
  if (WiFi.status() != WL_CONNECTED) return;

  if (activityChannelID == 0) {
    Serial.println("Enter your activity channel ID first.");
    return;
  }

  ThingSpeak.setField(1, predictedIndex);
  ThingSpeak.setField(2, predictedConfidence);
  ThingSpeak.setField(3, (long)completedWindows);

  int result = ThingSpeak.writeFields(
    activityChannelID,
    activityWriteKey
  );

  Serial.print("Activity ThingSpeak response: ");
  Serial.println(result);
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\nESP32 HEALTH MONITOR + TINYML");

  pinMode(ECG_LO_PLUS, INPUT);
  pinMode(ECG_LO_MINUS, INPUT);

  analogReadResolution(12);

  Wire.begin(21, 22);

  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("Health + TinyML");
  lcd.setCursor(0, 1);
  lcd.print("Starting...");

  sensors.begin();
  sensors.setWaitForConversion(false);
  sensors.requestTemperatures();
  temperatureRequestTime = millis();
  temperatureConversionPending = true;

  GPSserial.begin(9600, SERIAL_8N1, GPS_RX, GPS_TX);
  GSMserial.begin(9600, SERIAL_8N1, GSM_RX, GSM_TX);

  initMPU6050();

  if (!mpuAvailable) {
    Serial.println("WARNING: MPU unavailable.");
  }

  Serial.println("Loading activity model...");
  model = tflite::GetModel(activity_model);

  if (model == nullptr) {
    Serial.println("Model loading failed!");
    while (1) delay(1000);
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize
  );

  interpreter = &static_interpreter;

  Serial.println("Allocating tensors...");

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr ||
      input->type != kTfLiteInt8 ||
      output->type != kTfLiteInt8 ||
      input->dims->size != 3 ||
      input->dims->data[0] != 1 ||
      input->dims->data[1] != 128 ||
      input->dims->data[2] != 6 ||
      output->dims->size != 2 ||
      output->dims->data[0] != 1 ||
      output->dims->data[1] != 6) {
    Serial.println("Unexpected model tensor format!");
    while (1) delay(1000);
  }

  Serial.println("TinyML interpreter ready.");
  Serial.print("Input scale: ");
  Serial.println(input->params.scale, 8);
  Serial.print("Input zero point: ");
  Serial.println(input->params.zero_point);

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected");
  } else {
    Serial.println("WiFi connection failed");
  }

  ThingSpeak.begin(client);

  if (mpuAvailable) {
    Serial.println("Keep MPU6050 still for 5 seconds...");
    delay(5000);

    float ax, ay, az;
    float gx, gy, gz;

    if (readMPURegisters(ax, ay, az, gx, gy, gz)) {
      gravity[0] = ax;
      gravity[1] = ay;
      gravity[2] = az;
      gravityInitialized = true;
      Serial.println("Gravity initialized.");
    } else {
      Serial.println("Initial MPU reading failed.");
    }
  }

  lastMpuSampleUs = micros();

  lcd.clear();
  lcd.print("System Ready");

  Serial.println("Health monitoring and TinyML ready.");
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop() {
  readMPU6050();

  readPulse();
  readECG();
  readTemperature();
  readGPS();

  checkHealth();

  displayLCD();

  // Non-blocking SMS state machine
  servicePendingSMS();

  // Upload both channels every 20 seconds.
  // Note: HTTP/Wi-Fi operations can still interrupt the
  // 50 Hz MPU sampling loop while an upload is in progress.
  if (millis() - lastThingSpeak >= 20000UL) {
    lastThingSpeak = millis();

    updateHealthThingSpeak();
    updateActivityThingSpeak();
  }

  delay(1);
}
