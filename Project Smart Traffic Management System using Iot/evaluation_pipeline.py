import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score
)

# Metric helpers
def calculate_specificity(cm):
    """Specificity = TN / (TN + FP) for each class"""
    spec = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
        fp = cm[:,i].sum() - cm[i,i]
        spec.append(tn / (tn + fp) if (tn + fp) != 0 else 0)
    return spec

def calculate_sensitivity(cm):
    """Sensitivity (Recall) = TP / (TP + FN) for each class"""
    return [cm[i,i] / cm[i,:].sum() if cm[i,:].sum() else 0 for i in range(len(cm))]

def calculate_precision(cm):
    """Precision = TP / (TP + FP) for each class"""
    return [cm[i,i] / cm[:,i].sum() if cm[:,i].sum() else 0 for i in range(len(cm))]

# Evaluation block
def evaluate_and_report(df, source='Synthetic'):
    labels = ['Low', 'Medium', 'High']
    cm = confusion_matrix(df['true_category'], df['predicted_category'], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)

    print(f"\n{source} Classification Report:\n")
    print(classification_report(df['true_category'], df['predicted_category'], target_names=labels))

    acc = accuracy_score(df['true_category'], df['predicted_category'])
    sens = calculate_sensitivity(cm)
    spec = calculate_specificity(cm)
    prec = calculate_precision(cm)
    f1s = f1_score(df['true_category'], df['predicted_category'], average=None, labels=labels)

    # Assemble metric table
    metrics = pd.DataFrame({
        'Class': labels,
        'Accuracy (Overall)': [acc] * len(labels),
        'Precision': prec,
        'Recall (Sensitivity)': sens,
        'Specificity': spec,
        'F1 Score': f1s
    })

    print("\nMetrics per Class:")
    print(metrics.to_string(index=False))

    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{source} Confusion Matrix")
    plt.show()

# Main entry
if __name__ == "__main__":
    # Evaluate synthetic data
    df_syn = pd.read_csv('synthetic_predictions.csv')
    evaluate_and_report(df_syn, source='Synthetic')

    # Optional: Evaluate real Chicago dataset if exists
    try:
        df_chi = pd.read_csv('chicago_traffic.csv')
        from synthetic_pipeline import create_features, train_predictive_model
        df_chi_feat = create_features(df_chi)
        model = train_predictive_model(df_chi_feat)
        df_chi_feat['predicted_traffic_volume'] = model.predict(df_chi_feat[['hour','dayofweek','day','month','lag_1','lag_2','lag_3']])
        df_chi_feat['true_category'] = df_chi_feat['traffic_volume'].apply(lambda v: 'Low' if v < 200 else 'Medium' if v < 400 else 'High')
        df_chi_feat['predicted_category'] = df_chi_feat['predicted_traffic_volume'].apply(lambda v: 'Low' if v < 200 else 'Medium' if v < 400 else 'High')

        evaluate_and_report(df_chi_feat, source='Chicago')
    except FileNotFoundError:
        print("Chicago CSV not found; please add 'chicago_traffic.csv' and rerun.")
