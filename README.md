# Tunisian-Property-Price-Predictor
A machine learning-based web application for categorizing Tunisian property prices into low, mid, and high categories.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Tunisian Property Price Predictor is a machine learning-based application that predicts property price categories (Low, Mid, High) based on features such as the number of rooms, bathrooms, size, and region. This tool aims to enhance transparency in the Tunisian real estate market by providing both users and agencies with a reliable price estimation system.

## Features
- Predicts property price categories (Low, Mid, High).
- User-friendly interface for feature input and result visualization.
- Highlights feature importance in prediction.
- Interactive charts to display prediction confidence.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualizations

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mahdibenj22/Tunisian-Property-Price-Predictor.git

2. Navigate to the project directory:
   ```bash
   cd Tunisian-Property-Price-Predictor
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the application:
   ```bash
   streamlit run app.py

## File Structure
- `app.py`: Main application script.
- `models/xgb_classification_model.pkl`: Pre-trained XGBoost model for prediction.
- `data/`: Contains processed datasets.
- `visuals/`: Contains visual assets used in the application.

  

## Future Enhancements
- Expand dataset with more granular location details.
- Integrate advanced model ensemble techniques to improve accuracy.
- Add multi-language support for broader accessibility.
- Deploy on a cloud platform for global availability.


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Screenshots
**Input Features**
![Input Features](https://github.com/user-attachments/assets/59469a3c-9fc9-4b12-a839-906e03afd7ee)

**Prediction Results**
![Prediction Results](https://github.com/user-attachments/assets/10d29b82-4870-4055-b021-25a4bea65b5e)

**Feature Importance**
![Feature Importance](https://github.com/user-attachments/assets/1644aa60-d133-46b5-a852-cb663fbb4f24)
