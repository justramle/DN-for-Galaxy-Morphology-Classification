"""Shared plotting helpers for training curves and confusion matrices."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _plot_metric(ax, train_values, val_values, metric_name):
    ax.plot(train_values, label=f"Training {metric_name}")
    ax.plot(val_values, label=f"Validation {metric_name}")
    ax.set_title(f"Model {metric_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.legend(loc="upper left")


def plot_accuracy_and_loss(train_acc, val_acc, train_loss, val_loss, acc_filename, loss_filename):
    """Save accuracy and loss curves as two separate figures."""
    fig, ax = plt.subplots(figsize=(9, 4))
    _plot_metric(ax, train_acc, val_acc, "Accuracy")
    fig.savefig(acc_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    _plot_metric(ax, train_loss, val_loss, "Loss")
    fig.savefig(loss_filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_and_loss_combined(train_acc, val_acc, train_loss, val_loss, filename):
    """Save accuracy and loss curves side by side in one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    _plot_metric(ax1, train_acc, val_acc, "Accuracy")
    _plot_metric(ax2, train_loss, val_loss, "Loss")
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, filename, title=None, labels=None, cmap=None):
    cm = confusion_matrix(y_true, y_pred)
    kwargs = {"annot": True, "fmt": "d"}
    if cmap:
        kwargs["cmap"] = cmap
    if labels is not None:
        kwargs["xticklabels"] = labels
        kwargs["yticklabels"] = labels

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, **kwargs)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    if title:
        plt.title(title)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
