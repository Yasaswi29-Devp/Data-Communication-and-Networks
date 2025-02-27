# Data-Communication-and-Networks
# **Enhanced Network Security with Machine Learning-Based Intrusion Detection**

## **Project Overview**  
This project focuses on improving **network security** by developing and evaluating multiple **machine learning models** for **intrusion detection**. The system utilizes **various classification techniques** to detect and mitigate cyber threats effectively. The study emphasizes **data preprocessing, feature selection**, and **hyperparameter tuning** to enhance the accuracy of the models.

## **Key Features**  
🔹 **Machine Learning Models:** Decision Trees, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), Logistic Regression, Naïve Bayes.  
🔹 **Feature Selection:** Recursive Feature Elimination (RFE) with **Random Forest Importance**.  
🔹 **Hyperparameter Optimization:** Implemented using **Optuna** for KNN and Decision Trees.  
🔹 **Performance Evaluation:** Analyzed using metrics like **accuracy, precision, recall, and F1-score**.  
🔹 **Cross-Validation:** **10-fold cross-validation** to assess generalization performance.  
🔹 **Dataset Used:** 1999 DARPA Intrusion Detection Evaluation Dataset.  

## **Technologies Used**  
- **Programming Language:** Python  
- **Libraries:** Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn  
- **Optimization Toolkit:** Optuna  
- **Dataset Processing:** Recursive Feature Elimination (RFE), StandardScaler  

## **System Implementation**  
1. **Data Preprocessing**  
   - Handled missing values and removed irrelevant features.  
   - Scaled features using StandardScaler for consistency.  

2. **Model Training & Evaluation**  
   - Trained five different classification models.  
   - Evaluated performance using **accuracy, precision, recall, and F1-score**.  

3. **Hyperparameter Tuning**  
   - Used **Optuna** to optimize model parameters.  

4. **Cross-Validation**  
   - Applied **10-fold cross-validation** to validate model performance.  

## **Results & Performance**  
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|-----------|
| K-Nearest Neighbors (KNN) | 98% | 98% | 98% | 98% |
| Logistic Regression | 94% | 94% | 94% | 94% |
| Decision Tree       | 99% | 99% | 99% | 99% |
| Support Vector Classifier (SVC) | 97% | 97% | 97% | 97% |
| Naïve Bayes        | 89% | 90% | 89% | 89% |

**Conclusion:**  
- The **Decision Tree model** performed best, achieving **99% accuracy**.  
- **KNN and SVC models** also demonstrated strong generalization ability.  
- The study highlights the importance of **feature selection** and **hyperparameter tuning** in enhancing model performance.  

## **Future Enhancements**  
🚀 **Integrate Deep Learning** models for improved threat detection.  
🔐 **Real-Time Intrusion Detection** for live network monitoring.  
📊 **Dataset Expansion** by incorporating recent cyber-attack patterns.  

## **Contributors**  
| Name | Role |
|------|------|
| **Yasaswi Polasi** | Model Development & Research |
| **Dharma Reddy Pandem** | Data Preprocessing & Feature Selection |
| **Abhishek Medikonda** | Hyperparameter Tuning & Model Evaluation |
| **Sushma Ponugoti** | Performance Analysis & Report Writing |

## **References**  
1. Kostas, K. *Anomaly Detection in Networks Using Machine Learning*. Research Proposal, 2018.  
2. Alghanmi, N., Alotaibi, R., & Buhari, S. *Machine Learning Approaches for Anomaly Detection in IoT*. 2022.  
3. Ahmed, T., Oreshkin, B., & Coates, M. *Machine Learning Approaches to Network Anomaly Detection*. USENIX, 2007.  
