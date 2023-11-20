# StimulationB15

## Description

StimulationB15 is a project designed for the precise control of multiple hardware devices, such as cameras and LEDs, for scientific experimentation. It provides a wide range of features including signal synchronization, camera control, LED pattern generation, data logging, and a user-friendly graphical interface for streamlined operation.

## Features

- **Camera Control:** Capture video or images from multiple synchronized cameras.
- **TTL Signals:** Synchronize different hardware components with precision.
- **LED Patterns:** Generate custom LED sequences for experiments.
- **GUI Interface:** Easily control and configure experiments through a graphical user interface.
- **Data Logging:** Save and manage experimental data efficiently.

## File Structure

stimulationB15/
│
├── config.json
│
├── master_script.py
│
├── modules/
│ ├── camera_control.py
│ ├── data_logging.py
│ ├── led_pattern.py
│ ├── gui_interface.py
│ └── ttl_signals.py
│
├── utils/
│ ├── utility_functions.py
│
├── data/
│
├── .vscode/
│
└── README.md

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV
- Pypylon
- (Any other libraries or software required for your specific setup)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stimulationB15.git

pip install -r requirements.txt

### Usage

To start the program, run:

python master_script.py
Contributing

Please refer to CONTRIBUTING.md for information on how to contribute to this project.
License

This project is licensed under the MIT License. See the LICENSE.md file for details.
