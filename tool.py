import matplotlib.pyplot as plt

def plot_accuracy_loss(train_accuracies, test_accuracies, train_losses, save_dir):
    # 绘制精度-epoch折线图
    plt.figure()
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(save_dir + 'accuracy_epoch_plot.png')

    # 绘制loss-epoch折线图
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir + 'loss_epoch_plot.png')
