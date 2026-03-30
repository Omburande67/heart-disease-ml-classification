# Heart Disease Prediction

A machine learning web application for predicting heart disease risk using multiple classification models. This project implements a Streamlit-based interface that allows users to input health parameters and receive predictions from four different machine learning models.

## 🚀 Features

- **Multi-Model Prediction**: Compare predictions from Logistic Regression, Random Forest, Bagging Classifier, and XGBoost models
- **Interactive Web Interface**: User-friendly Streamlit application with custom styling
- **Real-time Input Validation**: Dynamic form inputs with appropriate constraints
- **Performance Metrics Display**: View accuracy, precision, recall, and F1 scores for each model
- **Comprehensive Health Parameters**: Input fields for all relevant health indicators

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning algorithms (Logistic Regression, Random Forest, Bagging)
- **XGBoost** - Gradient boosting framework
- **Pandas & NumPy** - Data manipulation and numerical operations
- **Imbalanced-learn (SMOTE)** - Handling class imbalance
- **Pickle** - Model serialization

## 📊 Dataset

The application uses the Framingham Heart Study dataset containing the following features:
- Gender, Age, Education
- Smoking status and cigarettes per day
- Blood pressure medication status
- Prevalent stroke and hypertension
- Diabetes status
- Total cholesterol, systolic/diastolic BP
- BMI, heart rate, glucose levels
- Target: Heart disease presence (Yes/No)

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.66 | 0.63 | 0.65 | 0.64 |
| Random Forest | 0.87 | 0.87 | 0.87 | 0.87 |
| Bagging Classifier | 0.80 | 0.80 | 0.82 | 0.81 |
| XGBoost | 0.88 | 0.88 | 0.89 | 0.88 |

## 📋 Prerequisites

- Python 3.7+
- Required Python packages (see requirements.txt)

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present:**
   - Place the trained model files in the project directory:
     - `Heart_disease_LR_model.pkl`
     - `Heart_disease_RF_model.pkl`
     - `Heart_disease_Bagging_model.pkl`
     - `Heart_disease_XGBoost_model.pkl`

4. **Add background image:**
   - Place `img2.jpeg` in the project directory for the app background

## 🚀 Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface:**
   - Open your browser and navigate to `http://localhost:8501`

3. **Make predictions:**
   - Fill in the health parameters in the form
   - Click the "Predict" button
   - View predictions from all four models
   - Review model performance metrics

## 📁 Project Structure

```
Heart_Disease_Prediction/
│
├── app.py                    # Main Streamlit application
├── LRBX.ipynb               # Jupyter notebook for model training
├── heart_disease.csv        # Dataset file
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── Heart_disease_LR_model.pkl       # Logistic Regression model
├── Heart_disease_RF_model.pkl       # Random Forest model
├── Heart_disease_Bagging_model.pkl  # Bagging Classifier model
├── Heart_disease_XGBoost_model.pkl  # XGBoost model
└── img2.jpeg               # Background image for the app
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Team Innovators**
- **College**: Vishwakarma Institute of Technology (VIT), Pune

## 🙏 Acknowledgments

- Framingham Heart Study for the dataset
- Scikit-learn and XGBoost communities for excellent ML libraries
- Streamlit for the amazing web app framework</content>
<parameter name="filePath">c:\Users\ombur\Desktop\BTECH (VIT)\EDI\Heart_Disease_Prediction\Heart_Disease_Prediction\README.md