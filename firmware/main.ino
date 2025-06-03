#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include <Wire.h>
#include "MAX30105.h"
#include "model_bp_data.h"  // BP estimation model data
#include "model_ecg_data.h"  // ECG model data
#include <Adafruit_MLX90614.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "qtable_sensors_time.h" // Q-table for the RL agent

// Constants for PPG signal
#define PPG_SIGNAL_SIZE 128
#define PPG_SAMPLING_FREQ 100  // Hz
#define PPG_SAMPLING_PERIOD_MS (1000 / PPG_SAMPLING_FREQ)
#define MOVING_AVG_SIZE 4

// Constants for ECG
#define ECG_SIGNAL_SIZE 180    // 500ms window at 360Hz
#define ECG_SAMPLING_FREQ 360  // Hz to match MIT-BIH database
#define ECG_SAMPLING_PERIOD_MS (1000 / ECG_SAMPLING_FREQ)
#define ECG_OUTPUT_PIN 14      // ADC pin for ECG
#define LO_PLUS 13            // ECG Lead-off detection positive
#define LO_MINUS 12           // ECG Lead-off detection negative
#define NORMAL_LED 4          // Green LED for normal rhythm
#define ABNORMAL_LED 5        // Red LED for abnormal rhythm
#define ERROR_LED 2           // Yellow LED for errors

// Pin definitions for ECG control
#define ECG_SDN_PIN 15        // Shutdown pin for AD8232

// Pin definitions for I2C
#define I2C_SDA 8         // I2C Data pin
#define I2C_SCL 9         // I2C Clock pin

// I2C Addresses (default addresses for both sensors)
#define MAX30105_I2C_ADDR 0x57
#define MLX90614_I2C_ADDR 0x5A

// Global variables for PPG
MAX30105 particleSensor;
float ppg_buffer[PPG_SIGNAL_SIZE];
uint32_t moving_avg_buffer[MOVING_AVG_SIZE];
int ppg_buffer_index = 0;
int moving_avg_index = 0;
float ppg_min = 50000;  // Dynamic range adjustment
float ppg_max = 0;      // Dynamic range adjustment

// Global variables for ECG
float ecg_buffer[ECG_SIGNAL_SIZE];
int ecg_buffer_index = 0;
const char* ECG_LABELS[] = {"Normal", "Left BBB", "Right BBB", "Atrial Premature", "PVC"};

// MLX90614 temperature sensor
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// TFLite globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

// BP estimation model
extern const unsigned char model_data[];  // From BP estimation/model_data.h
const tflite::Model* ppg_model = nullptr;
tflite::MicroInterpreter* ppg_interpreter = nullptr;
TfLiteTensor* ppg_input = nullptr;
TfLiteTensor* ppg_output = nullptr;

// ECG classification model
extern const unsigned char model_tflite[];  // From ECG/model_data.h
extern const unsigned int model_tflite_len;
const tflite::Model* ecg_model = nullptr;
tflite::MicroInterpreter* ecg_interpreter = nullptr;
TfLiteTensor* ecg_input = nullptr;
TfLiteTensor* ecg_output = nullptr;

// Create separate tensor arenas for each model
constexpr int kTensorArenaSize = 32768;
uint8_t ppg_tensor_arena[kTensorArenaSize];
uint8_t ecg_tensor_arena[kTensorArenaSize];

// Clinical thresholds for ECG metrics
const float MIN_QRS_DURATION = 80.0;  // ms
const float MAX_QRS_DURATION = 120.0; // ms
const float MIN_ST_ELEVATION = -0.1;  // mV
const float MAX_ST_ELEVATION = 0.2;   // mV
const float MIN_HEART_RATE = 40.0;    // BPM
const float MAX_HEART_RATE = 200.0;   // BPM

// Structure to hold ECG metrics with validation
struct ECGMetrics {
    float heart_rate;
    const char* rhythm_class;
    float qrs_duration;
    float st_elevation;
    bool valid_hr;
    bool valid_qrs;
    bool valid_st;
    float confidence_score;
};

// Buffer for R-peak detection
const int MAX_PEAKS = 10;
float r_peak_timestamps[MAX_PEAKS];
int peak_count = 0;
unsigned long last_peak_time = 0;
float last_r_peak_value = 0;

// RL agent state variables
int battery_level = 10;         // Start with full battery (0-10)
int current_time_step = 0;      // Current time step
bool arr_flag = false;          // Arrhythmia detected flag
bool bp_flag = false;           // Abnormal blood pressure flag
bool fever_flag = false;        // Fever detected flag
int current_action = 7;         // Default action: all sensors on

// Sensor state variables
bool ecg_enabled = true;        // ECG sensor state
bool ppg_enabled = true;        // PPG sensor state
bool temp_enabled = true;       // Temperature sensor state

// Battery simulation
const unsigned long BATTERY_UPDATE_INTERVAL = 5000;  // 1 minute
unsigned long last_battery_update = 0;
const float BATTERY_DRAIN_RATE = 0.1;                // Base drain rate per update

// Thresholds for R-peak detection
const float PEAK_THRESHOLD = 0.7;  // Normalized threshold
const int MIN_RR_SAMPLES = 108;    // Minimum 300ms at 360Hz

// Function declarations for PPG processing
uint32_t calculate_moving_average(uint32_t new_value) {
    uint32_t sum = 0;
    moving_avg_buffer[moving_avg_index] = new_value;
    moving_avg_index = (moving_avg_index + 1) % MOVING_AVG_SIZE;
    
    for (int i = 0; i < MOVING_AVG_SIZE; i++) {
        sum += moving_avg_buffer[i];
    }
    
    return sum / MOVING_AVG_SIZE;
}

float normalize_ppg(uint32_t value) {
    // Update min/max values
    if (value < ppg_min) ppg_min = value;
    if (value > ppg_max) ppg_max = value;
    
    // Prevent division by zero
    if (ppg_max == ppg_min) return 0.5f;
    
    // Normalize to 0-1 range
    return (float)(value - ppg_min) / (float)(ppg_max - ppg_min);
}

// Function declarations for ECG processing
void normalize_ecg() {
    float mean = 0.0;
    float std = 0.0;
    
    // Calculate mean
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        mean += ecg_buffer[i];
    }
    mean /= ECG_SIGNAL_SIZE;
    
    // Calculate standard deviation
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        std += (ecg_buffer[i] - mean) * (ecg_buffer[i] - mean);
    }
    std = sqrt(std / ECG_SIGNAL_SIZE);
    
    // Normalize the data
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        ecg_buffer[i] = (ecg_buffer[i] - mean) / (std + 1e-6);
    }
}

// Detect R-peaks in the signal
bool detectRPeak(float current_sample, unsigned long current_time) {
    static float max_value = 0;
    static int samples_since_last_peak = 0;
    
    samples_since_last_peak++;
    
    // Update maximum value
    if (current_sample > max_value) {
        max_value = current_sample;
    }
    
    // Check if this is a peak
    if (current_sample > PEAK_THRESHOLD && 
        current_sample > last_r_peak_value && 
        samples_since_last_peak >= MIN_RR_SAMPLES) {
        
        // Store peak information
        if (peak_count < MAX_PEAKS) {
            r_peak_timestamps[peak_count++] = current_time;
        } else {
            // Shift array and add new peak
            for (int i = 0; i < MAX_PEAKS - 1; i++) {
                r_peak_timestamps[i] = r_peak_timestamps[i + 1];
            }
            r_peak_timestamps[MAX_PEAKS - 1] = current_time;
        }
        
        last_peak_time = current_time;
        last_r_peak_value = current_sample;
        samples_since_last_peak = 0;
        max_value = 0;
        return true;
    }
    
    return false;
}

// Calculate heart rate from R-peak intervals
float calculateHeartRate() {
    if (peak_count < 2) return 0;
    
    float total_interval = 0;
    int intervals = 0;
    
    // Calculate average interval between peaks
    for (int i = 1; i < peak_count; i++) {
        float interval = (r_peak_timestamps[i] - r_peak_timestamps[i-1]) / 1000.0; // Convert to seconds
        total_interval += interval;
        intervals++;
    }
    
    if (intervals == 0) return 0;
    
    float avg_interval = total_interval / intervals;
    return 60.0 / avg_interval; // Convert to BPM
}

// Improved QRS duration measurement using Pan-Tompkins approach
float measureQRSDuration(float* signal, int peak_index) {
    // Parameters for QRS detection
    const float HIGH_PASS_ALPHA = 0.95;  // High-pass filter coefficient
    const float LOW_PASS_ALPHA = 0.15;   // Low-pass filter coefficient
    
    // Buffers for filtered signals
    float filtered[ECG_SIGNAL_SIZE];
    float derivative[ECG_SIGNAL_SIZE];
    float squared[ECG_SIGNAL_SIZE];
    
    // High-pass filter
    for (int i = 1; i < ECG_SIGNAL_SIZE; i++) {
        filtered[i] = HIGH_PASS_ALPHA * (filtered[i-1] + signal[i] - signal[i-1]);
    }
    
    // Low-pass filter
    for (int i = 1; i < ECG_SIGNAL_SIZE; i++) {
        filtered[i] = filtered[i] + LOW_PASS_ALPHA * (filtered[i-1] - filtered[i]);
    }
    
    // Derivative
    for (int i = 2; i < ECG_SIGNAL_SIZE-2; i++) {
        derivative[i] = (2*filtered[i+2] + filtered[i+1] - filtered[i-1] - 2*filtered[i-2]) / 8.0;
    }
    
    // Square
    for (int i = 0; i < ECG_SIGNAL_SIZE; i++) {
        squared[i] = derivative[i] * derivative[i];
    }
    
    // Find QRS onset and offset using adaptive threshold
    float max_energy = 0;
    for (int i = peak_index-20; i <= peak_index+20 && i < ECG_SIGNAL_SIZE; i++) {
        if (i >= 0 && squared[i] > max_energy) {
            max_energy = squared[i];
        }
    }
    
    float threshold = max_energy * 0.15;  // 15% of max energy
    
    // Find QRS onset
    int onset = peak_index;
    for (int i = peak_index; i >= 0 && i > peak_index - 40; i--) {
        if (squared[i] < threshold) {
            onset = i;
            break;
        }
    }
    
    // Find QRS offset
    int offset = peak_index;
    for (int i = peak_index; i < ECG_SIGNAL_SIZE && i < peak_index + 40; i++) {
        if (squared[i] < threshold) {
            offset = i;
            break;
        }
    }
    
    // Convert samples to milliseconds
    return (offset - onset) * (1000.0 / ECG_SAMPLING_FREQ);
}

// Improved ST segment measurement with baseline correction
float measureSTElevation(float* signal, int peak_index) {
    // Parameters
    const int PR_SEGMENT_START = 50;  // samples before R peak
    const int ST_SEGMENT_START = 80;  // samples after R peak
    const int SEGMENT_LENGTH = 10;    // samples to average
    
    float pr_baseline = 0;
    float st_level = 0;
    int pr_samples = 0;
    int st_samples = 0;
    
    // Calculate PR baseline (reference level)
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        int idx = peak_index - PR_SEGMENT_START + i;
        if (idx >= 0 && idx < ECG_SIGNAL_SIZE) {
            pr_baseline += signal[idx];
            pr_samples++;
        }
    }
    
    // Calculate ST level
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        int idx = peak_index + ST_SEGMENT_START + i;
        if (idx >= 0 && idx < ECG_SIGNAL_SIZE) {
            st_level += signal[idx];
            st_samples++;
        }
    }
    
    // Avoid division by zero
    if (pr_samples == 0 || st_samples == 0) return 0;
    
    pr_baseline /= pr_samples;
    st_level /= st_samples;
    
    // Return ST deviation from baseline
    return st_level - pr_baseline;
}

// Format metrics as JSON
void formatMetricsJSON(ECGMetrics metrics) {
    Serial.print("{\"ecg_metrics\":{");
    Serial.print("\"heart_rate\":");
    Serial.print(metrics.heart_rate);
    Serial.print(",\"rhythm_class\":\"");
    Serial.print(metrics.rhythm_class);
    Serial.print("\",\"qrs_duration\":");
    Serial.print(metrics.qrs_duration);
    Serial.print(",\"st_elevation\":");
    Serial.print(metrics.st_elevation);
    Serial.print(",\"valid_hr\":");
    Serial.print(metrics.valid_hr ? "true" : "false");
    Serial.print(",\"valid_qrs\":");
    Serial.print(metrics.valid_qrs ? "true" : "false");
    Serial.print(",\"valid_st\":");
    Serial.print(metrics.valid_st ? "true" : "false");
    Serial.print(",\"confidence_score\":");