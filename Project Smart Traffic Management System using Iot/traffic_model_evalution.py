import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
)

def calculate_metrics(conf_matrix):
    metrics = {
        'TP': [],
        'FP': [],
        'TN': [],
        'FN': [],
        'Precision': [],
        'Recall (Sensitivity)': [],
        'Specificity': [],
        'F1 Score': [],
        'Class Accuracy': [],
    }

    total = conf_matrix.sum()
    num_classes = len(conf_matrix)

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        TN = total - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        class_accuracy = (TP + TN) / total if total != 0 else 0

        metrics['TP'].append(TP)
        metrics['FP'].append(FP)
        metrics['TN'].append(TN)
        metrics['FN'].append(FN)
        metrics['Precision'].append(precision)
        metrics['Recall (Sensitivity)'].append(recall)
        metrics['Specificity'].append(specificity)
        metrics['F1 Score'].append(f1)
        metrics['Class Accuracy'].append(class_accuracy)

    return metrics

if __name__ == "__main__":
    # Load saved predictions
    df = pd.read_csv('traffic_predictions.csv')

    y_true = df['true_category']
    y_pred = df['predicted_category']

    labels = ['Low', 'Medium', 'High']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    overall_accuracy = accuracy_score(y_true, y_pred)
    metrics = calculate_metrics(cm)

    metrics_df = pd.DataFrame({
        'Class': labels,
        'TP': metrics['TP'],
        'FP': metrics['FP'],
        'TN': metrics['TN'],
        'FN': metrics['FN'],
        'Class Accuracy': metrics['Class Accuracy'],
        'Precision': metrics['Precision'],
        'Sensitivity (Recall)': metrics['Recall (Sensitivity)'],
        'Specificity': metrics['Specificity'],
        'F1 Score': metrics['F1 Score'],
        'Overall Accuracy': [overall_accuracy] * len(labels),
    })

    print("\nMetrics per Class:")
    print(metrics_df.to_string(index=False))

    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Traffic Volume Classification")
    plt.show()
    