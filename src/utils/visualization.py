import json
import matplotlib.pyplot as plt

def plot_metrics(metrics_file):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(metrics["loss"], label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, output_cm_file="confusion_matrix.png"):
    """
    Plot and save the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_cm_file)
    print(f"Confusion matrix plot saved to {output_cm_file}")
    plt.show()
    
if __name__ == "__main__":
    plot_metrics("training_metrics.json")