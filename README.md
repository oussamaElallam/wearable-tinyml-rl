# Wearable TinyML Reinforcement Learning System

This repository contains the code, firmware, training scripts, and hardware setup for a wearable health monitoring system using TinyML and reinforcement learning on the ESP32-S3 platform.

## Directory Structure

```
wearable-tinyml-rl/
├── firmware/
│   ├── main.ino                # Main firmware for ESP32-S3
│   ├── bp_estimator.tflite     # Blood pressure estimation model (binary, large)
│   ├── ecg_classifier.tflite   # ECG arrhythmia classification model (binary, large)
│   └── qtable_alpha12.bin      # Q-table for RL agent (binary, large)
├── hardware_setup.md           # Hardware setup and wiring instructions
├── training/
│   ├── train_q_learning.py     # Q-learning training script
│   └── synthetic_evaluation.py # Synthetic evaluation script
└── README.md                   # Project overview and instructions
```

## Requirements

- ESP32-S3 development board
- Arduino IDE (with ESP32 board support)
- Python 3.7+
- TensorFlow Lite for Microcontrollers
- Sensors: MAX30105 (PPG), AD8232 (ECG), MLX90614 (temperature)

## Installation

### Firmware

1. Open `firmware/main.ino` in Arduino IDE.
2. Install required libraries: `MAX30105`, `Adafruit_MLX90614`, `TensorFlowLite_ESP32`.
3. Select the ESP32-S3 board and upload the firmware.
4. Place model files (`bp_estimator.tflite`, `ecg_classifier.tflite`) and Q-table (`qtable_alpha12.bin`) in the `firmware/` directory.

### Python Scripts

- Install Python dependencies:
  ```bash
  pip install numpy
  ```
- Run training:
  ```bash
  python training/train_q_learning.py
  ```
- Run evaluation:
  ```bash
  python training/synthetic_evaluation.py
  ```

## Training and Evaluation

- `train_q_learning.py` trains a Q-learning agent to optimize sensor usage and battery life.
- `synthetic_evaluation.py` evaluates different sensor policies (RL, always-on, periodic) on synthetic patient data.

## Hardware Setup

See `hardware_setup.md` for wiring diagrams and setup instructions for ESP32-S3, PPG, ECG, and temperature sensors.

## Model Files

- `bp_estimator.tflite`: Blood pressure estimation model (TensorFlow Lite)
- `ecg_classifier.tflite`: ECG arrhythmia classification model (TensorFlow Lite)
- `qtable_alpha12.bin`: Q-table for RL agent

**Note:** These files are large and may require Git LFS or manual upload if pushing to GitHub.

## License

MIT License. See LICENSE file for details.
