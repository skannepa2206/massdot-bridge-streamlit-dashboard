# MassDOT Chelsea Bridge Lift Prediction Dashboard

## 🌉 Overview

This is an AI-powered dashboard for predicting and managing Chelsea Bridge lift operations. The system uses advanced machine learning models (MLP and TabNet) to predict bridge lift start times and durations, helping travelers and operators plan efficiently.

## 🚀 Features

### 🤖 **AI Predictions**
- **Start Time Prediction**: 87% accuracy using MLP neural network
- **Duration Prediction**: 71% accuracy using TabNet model
- **Full 24-hour coverage**: Predictions from 00:00 to 23:59
- **Weather integration**: Real-time weather data affects predictions
- **Tidal modeling**: Accurate tidal calculations influence lift timing

### 📊 **Real-time Dashboard**
- **Live predictions**: Daily bridge lift schedules
- **Historical data**: View actual bridge logs when available
- **Performance metrics**: KPI cards showing daily statistics
- **Interactive charts**: Analytics and trend visualization
- **Responsive design**: Works on desktop and mobile

### 🔧 **Admin Features**
- **Data management**: Upload new bridge logs
- **Smart integration**: Date-based data merging
- **Communication tools**: X (Twitter) and VMS integration
- **System monitoring**: Health checks and performance metrics

## 📋 Prerequisites

### **Required Python Packages**
```
streamlit
pandas
numpy
joblib
requests
matplotlib
plotly
openpyxl
scikit-learn
torch
pytz
```

### **Required Files**
- `models/mlp_model.pkl` - Trained MLP model for start time prediction
- `models/tabnet_model.pkl` - Trained TabNet model for duration prediction
- `models/scaler.pkl` - Feature scaler for MLP model
- `models/features_used.pkl` - List of features used during training

### **Optional Data Files**
- `data/enriched_bridge_data.csv` - Historical bridge data
- `data/bridge_logs_master.xlsx` - Alternative data format

## 🛠️ Installation

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd massdot-bridge-dashboard
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Prepare Model Files**
Ensure your trained models are in the `models/` directory:
- Train your MLP and TabNet models
- Save them using `joblib.dump(model, 'models/model_name.pkl')`
- Ensure scaler and features list are also saved

### **4. Run the Dashboard**
```bash
streamlit run enhanced_dashboard.py
```

## 📖 User Guide

### **🏠 Main Dashboard**

#### **Status Banner**
- **Green**: No lifts predicted - clear travel
- **Orange**: Few lifts predicted - plan accordingly  
- **Pink**: Many lifts predicted - expect delays

#### **KPI Cards**
- **Predicted Lifts**: Total number of lifts for the day
- **Avg Duration**: Average duration of predicted lifts
- **Next Lift**: Time of next upcoming lift
- **Weather**: Current temperature

#### **Prediction Table**
Shows detailed schedule with:
- **Lift Number**: Sequential numbering
- **Start Time**: Predicted start time (HH:MM)
- **End Time**: Calculated end time
- **Duration**: Expected duration in minutes

### **📅 Date Selection**

#### **Regular Users**
- Select dates up to **7 days** in the future
- View predictions for planning trips
- Check historical data when available

#### **Admin Users**
- Access up to **30 days** of predictions
- Extended forecasting capabilities
- Data management features

### **🔐 Admin Access**

#### **Login**
1. In the sidebar, find "Admin Access"
2. Enter password: `MassDOT2025!`
3. Click "Login"

#### **Admin Features**

##### **📢 Communication Tools**
- **X (Twitter) Integration**:
  - Auto-generates bridge schedule posts
  - Click "SEND TO X" to post directly
  - Follows MassDOT format standards

- **VMS (Variable Message Signs)**:
  - Creates highway sign messages
  - Click "SEND TO VMS" to broadcast
  - Covers 3 VMS locations

##### **📊 Data Management**
- **Upload New Bridge Logs**:
  - Supports Excel (.xlsx) and CSV files
  - Required format: `Start Time, End Time, Duration, Direction, Vessel(s)`
  - Example: `6/1/2023 0:22, 6/1/2023 0:40, 0:18, OUT, Justice/Gracie`

- **Smart Data Integration**:
  - **Existing dates**: Replaces old data with new uploads
  - **New dates**: Adds to existing dataset
  - **Automatic formatting**: Standardizes column formats
  - **Immediate updates**: UI reflects changes instantly

##### **📈 Analytics Dashboard**
- **Performance Metrics**: Model accuracy trends
- **Traffic Patterns**: Daily and hourly analysis
- **Data Quality**: Statistics and health checks
- **System Status**: Component monitoring

## 📄 File Formats

### **Upload Format**
Your bridge log files should have these columns:
```
Start Time, End Time, Duration, Direction, Vessel(s)
```

#### **Example Data**
```
Start Time,End Time,Duration,Direction,Vessel(s)
6/1/2023 0:22,6/1/2023 0:40,18 min,OUT,Justice/Gracie Reinauer
6/1/2023 3:11,6/1/2023 3:24,13 min,IN,RTC 109
6/1/2023 5:10,6/1/2023 5:22,12 min,OUT,Container Ship
```

#### **Date Formats Supported**
- `MM/DD/YYYY HH:MM` (e.g., 6/1/2023 14:30)
- `YYYY-MM-DD HH:MM` (e.g., 2023-06-01 14:30)
- `DD/MM/YYYY HH:MM` (auto-detected)

#### **Duration Formats**
- `XX min` (e.g., 15 min)
- `XX minutes` (e.g., 15 minutes)
- Numeric values (automatically converted)

## 🔧 Troubleshooting

### **Common Issues**

#### **"Missing Models" Error**
```
🔴 Critical Error: Required Models Missing
```
**Solution**: 
1. Ensure all 4 model files exist in `models/` directory
2. Check files are not empty (0MB)
3. Retrain and save models if necessary

#### **"openpyxl" Error**
```
❌ Missing optional dependency 'openpyxl'
```
**Solution**:
```bash
pip install openpyxl
```
Or use CSV files instead of Excel.

#### **"Array Conversion" Error**
```
❌ only length-1 arrays can be converted to Python scalars
```
**Solution**: This is handled automatically in the latest version. Restart the dashboard.

#### **Empty Predictions**
**Possible Causes**:
- Models not trained properly
- Feature mismatch between training and prediction
- Invalid date selection

**Solution**:
1. Check model training data
2. Verify feature engineering matches training
3. Contact system administrator

### **Performance Issues**

#### **Slow Loading**
- Check internet connection (weather API calls)
- Verify model file sizes
- Clear browser cache

#### **Memory Issues**
- Large datasets may cause slowdowns
- Consider data archiving for old records
- Monitor system resources

## 🎯 Best Practices

### **For Regular Users**
1. **Check predictions before traveling** during peak hours (7-10 AM, 4-7 PM)
2. **Allow extra time** when multiple lifts are predicted
3. **Monitor weather conditions** - they significantly impact accuracy
4. **Use mobile-friendly interface** for on-the-go checking

### **For Operators**
1. **Update bridge logs regularly** to improve model accuracy
2. **Use communication tools** to keep public informed
3. **Monitor system health** in admin dashboard
4. **Archive old data** to maintain performance

### **For Data Managers**
1. **Maintain consistent format** when uploading data
2. **Validate data quality** before integration
3. **Back up existing data** before major uploads
4. **Monitor prediction accuracy** and retrain models when needed

## 📞 Support

### **Technical Issues**
- Check this README first
- Review error messages carefully
- Verify all dependencies are installed

### **Data Questions**
- Ensure proper file format
- Check date ranges and formatting
- Validate required columns exist

### **Feature Requests**
- Document specific requirements
- Consider impact on existing functionality
- Test thoroughly before deployment

## 🔄 Updates and Maintenance

### **Regular Maintenance**
1. **Model Retraining**: Quarterly or when accuracy drops
2. **Data Cleanup**: Monthly archiving of old records
3. **System Updates**: Keep dependencies current
4. **Performance Monitoring**: Weekly health checks

### **Backup Procedures**
1. **Model Files**: Back up trained models regularly
2. **Historical Data**: Archive bridge logs monthly
3. **Configuration**: Save dashboard settings
4. **System State**: Document current setup

## 📊 System Architecture

### **Components**
- **Frontend**: Streamlit web interface
- **ML Models**: MLP (start time) + TabNet (duration)
- **Data Layer**: CSV/Excel file storage
- **APIs**: Weather data integration
- **Communication**: X/Twitter and VMS integration

### **Data Flow**
1. **Input**: Historical bridge logs
2. **Processing**: Feature engineering and model training
3. **Prediction**: Real-time ML inference
4. **Output**: Dashboard predictions and communications
5. **Feedback**: New data integration and model updates

## 🚀 Future Enhancements

### **Planned Features**
- **Real-time vessel tracking** integration
- **Advanced weather models** for better accuracy
- **Mobile app** development
- **API endpoints** for external integration
- **Machine learning automation** for continuous improvement

### **Performance Improvements**
- **Caching strategies** for faster loading
- **Database backend** for better data management
- **Load balancing** for high traffic
- **Automated monitoring** and alerting

---

## 📄 License

This project is developed for MassDOT operations. All rights reserved.

## 🤝 Contributing

For contributions or improvements, please contact the development team with detailed proposals and testing results.

---

**Version**: 1.0  
**Last Updated**: June 2025  
**Developed for**: Massachusetts Department of Transportation (MassDOT)
