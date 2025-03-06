
# Predictive Maintenance System for Manufacturing

### **Problem Description**
Manufacturing businesses face high costs due to unplanned downtimes caused by mechanical failures. Predictive maintenance aims to forecast these failures in advance to minimize downtime and costs. This project develops a predictive model to answer: **"What is the probability that a machine will fail soon due to a specific component failure?"** The problem is framed as a multi-class classification task using historical machine data.

---

### **Data Sources**
The dataset consists of:
1. **Telemetry:** Hourly data on voltage, rotation, pressure, and vibration for 100 machines.
2. **Errors:** Logs of non-breaking errors with timestamps.
3. **Maintenance:** Records of both scheduled and unscheduled maintenance.
4. **Failures:** Data on component failures with timestamps.
5. **Machines:** Machine details including model type and age.

---

### **Data Loading and Preparation**
```python
import pandas as pd

# Load datasets
telemetry = pd.read_csv('PdM_telemetry.csv')
errors = pd.read_csv('PdM_errors.csv')
maint = pd.read_csv('PdM_maint.csv')
failures = pd.read_csv('PdM_failures.csv')
machines = pd.read_csv('PdM_machines.csv')

# Convert datetime columns
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
errors['datetime'] = pd.to_datetime(errors['datetime'])
maint['datetime'] = pd.to_datetime(maint['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])
```

---

### **Feature Engineering**
1. **Telemetry:** Created rolling mean and standard deviation for 3-hour and 24-hour windows.
2. **Errors:** Transformed error IDs into categorical counts per hour.
3. **Maintenance:** Processed maintenance logs to identify component replacements.
4. **Failures:** Labeled data based on component failure types.

---

### **Exploratory Data Analysis**
- Analyzed distributions of telemetry data.
- Plotted error types and frequency.
- Examined maintenance and failure rates per component.

---

### **Model Development**
- **Algorithm:** Used machine learning classification algorithms.
- **Features:** Combined telemetry stats, error counts, and maintenance history.
- **Evaluation:** Assessed model accuracy, precision, and recall.

---

### **Conclusion**
The predictive model provides actionable insights for preemptive maintenance, reducing downtime and costs in manufacturing.
