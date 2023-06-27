# @Time : 2023/6/11 19:36
# @Author : yichen
import re
import matplotlib.pyplot as plt
import os



def displayPrecisionMetrics():
    val_interval = ALLepoch[0] / len(train_precision_class0)
    epochs = range(1, len(train_precision_class0) + 1)
    epochs = [val_interval * i for i in epochs]
    ncol=2

    train_class0_label = f'train_class0 ({train_precision_class0[-1]:.2f})'
    val_class0_label = f'val_class0 ({val_precision_class0[-1]:.2f})'
    test_class0_label = f'test_class0 ({test_precision_class0[-1]:.2f})'

    train_class1_label = f'train_class1 ({train_precision_class1[-1]:.2f})'
    val_class1_label = f'val_class1 ({val_precision_class1[-1]:.2f})'
    test_class1_label = f'test_class1 ({test_precision_class1[-1]:.2f})'

    plt.plot(epochs, train_precision_class0, 'b-', label=train_class0_label)
    plt.plot(epochs, val_precision_class0, 'm-', label=val_class0_label)
    plt.plot(epochs, test_precision_class0, 'c-', label=test_class0_label)

    plt.plot(epochs, train_precision_class1, 'b--', label=train_class1_label)
    plt.plot(epochs, val_precision_class1, 'm--', label=val_class1_label)
    plt.plot(epochs, test_precision_class1, 'c--',  label=test_class1_label)

    if task == 'T2':
        train_class2_label = f'train_class2 ({train_precision_class2[-1]:.2f})'
        val_class2_label = f'val_class2 ({val_precision_class2[-1]:.2f})'
        test_class2_label = f'test_class2 ({test_precision_class2[-1]:.2f})'
        plt.plot(epochs, train_precision_class2, 'b:', label=train_class2_label)
        plt.plot(epochs, val_precision_class2, 'm:', label=val_class2_label)
        plt.plot(epochs, test_precision_class2, 'c:', label=test_class2_label)
        ncol =3
    # 添加图例、坐标轴标签和标题
    plt.legend(loc='lower left', ncol=ncol)
    plt.xlabel('Epochs')
    plt.ylabel('Presicion Value')
    plt.title(task + ' Presicion Metrics')
    plt.ylim(0, 110)
    plt.show()


def displayRecallMetrics():
    val_interval = ALLepoch[0] / len(train_recall_class0)
    epochs = range(1, len(train_recall_class0) + 1)
    epochs = [val_interval * i for i in epochs]
    ncol = 2

    train_class0_label = f'train_class0 ({train_recall_class0[-1]:.2f})'
    val_class0_label = f'val_class0 ({val_recall_class0[-1]:.2f})'
    test_class0_label = f'test_class0 ({test_recall_class0[-1]:.2f})'

    train_class1_label = f'train_class1 ({train_recall_class1[-1]:.2f})'
    val_class1_label = f'val_class1 ({val_recall_class1[-1]:.2f})'
    test_class1_label = f'test_class1 ({test_recall_class1[-1]:.2f})'

    plt.plot(epochs, train_recall_class0, 'b-', label=train_class0_label)
    plt.plot(epochs, val_recall_class0, 'm-', label=val_class0_label)
    plt.plot(epochs, test_recall_class0, 'c-', label=test_class0_label)

    plt.plot(epochs, train_recall_class1, 'b--', label=train_class1_label)
    plt.plot(epochs, val_recall_class1, 'm--', label=val_class1_label)
    plt.plot(epochs, test_recall_class1, 'c--', label=test_class1_label)

    if task == 'T2':
        train_class2_label = f'train_class2 ({train_recall_class2[-1]:.2f})'
        val_class2_label = f'val_class2 ({val_recall_class2[-1]:.2f})'
        test_class2_label = f'test_class2 ({test_recall_class2[-1]:.2f})'

        plt.plot(epochs, train_recall_class2, 'b:', label=train_class2_label)
        plt.plot(epochs, val_recall_class2, 'm:', label=val_class2_label)
        plt.plot(epochs, test_recall_class2, 'c:', label=test_class2_label)
        ncol = 3
    # 添加图例、坐标轴标签和标题
    plt.legend(loc='lower left', ncol=ncol)
    plt.xlabel('Epochs')
    plt.ylabel('Recall Value')
    plt.title(task + ' Recall Metrics')
    plt.ylim(0, 110)
    plt.show()


def displayFscoreMetrics():
    val_interval = ALLepoch[0] / len(train_f_score_class0)
    epochs = range(1, len(train_f_score_class0) + 1)
    epochs = [val_interval * i for i in epochs]
    ncol = 2

    train_class0_label = f'train_class0 ({train_f_score_class0[-1]:.2f})'
    val_class0_label = f'val_class0 ({val_f_score_class0[-1]:.2f})'
    test_class0_label = f'test_class0 ({test_f_score_class0[-1]:.2f})'

    train_class1_label = f'train_class1 ({train_f_score_class1[-1]:.2f})'
    val_class1_label = f'val_class1 ({val_f_score_class1[-1]:.2f})'
    test_class1_label = f'test_class1 ({test_f_score_class1[-1]:.2f})'

    plt.plot(epochs, train_f_score_class0, 'b-', label=train_class0_label)
    plt.plot(epochs, val_f_score_class0, 'm-', label=val_class0_label)
    plt.plot(epochs, test_f_score_class0, 'c-', label=test_class0_label)

    plt.plot(epochs, train_f_score_class1, 'b--', label=train_class1_label)
    plt.plot(epochs, val_f_score_class1, 'm--', label=val_class1_label)
    plt.plot(epochs, test_f_score_class1, 'c--', label=test_class1_label)

    if task == 'T2':
        train_class2_label = f'train_class2 ({train_f_score_class2[-1]:.2f})'
        val_class2_label = f'val_class2 ({val_f_score_class2[-1]:.2f})'
        test_class2_label = f'test_class2 ({test_f_score_class2[-1]:.2f})'

        plt.plot(epochs, train_f_score_class2, 'b:', label=train_class2_label)
        plt.plot(epochs, val_f_score_class2, 'm:', label=val_class2_label)
        plt.plot(epochs, test_f_score_class2, 'c:', label=test_class2_label)
        ncol = 3
    # 添加图例、坐标轴标签和标题
    plt.legend(loc='lower left', ncol=ncol)
    plt.xlabel('Epochs')
    plt.ylabel('F-score Value')
    plt.title(task + ' F-score Metrics')
    plt.ylim(0, 110)
    plt.show()


def displayKappaMetrics():
    val_interval = ALLepoch[0] / len(train_kappa)
    epochs = range(1, len(train_kappa) + 1)
    epochs = [val_interval * i for i in epochs]

    train_kappa_label = f'train ({train_kappa[-1]:.2f})'
    val_kappa_label = f'val ({val_kappa[-1]:.2f})'
    test_kappa_label = f'test ({test_kappa[-1]:.2f})'

    plt.plot(epochs, train_kappa, 'b-', label=train_kappa_label)
    plt.plot(epochs, val_kappa, 'm-', label=val_kappa_label)
    plt.plot(epochs, test_kappa, 'c-', label=test_kappa_label)
    # 添加图例、坐标轴标签和标题
    plt.legend(loc='lower left')
    plt.xlabel('Epochs')
    plt.ylabel('Kappa Value')
    plt.title(task + ' Kappa Metrics')
    plt.ylim(0, 110)
    plt.show()


def drawLoss():
    interval = ALLepoch[0] / len(loss_value)
    xrange = range(1, len(loss_value) + 1)
    xrange = [interval * i for i in xrange]
    plt.plot(xrange, loss_value, 'c-', label='loss')  # 红色线条
    plt.legend(loc='lower left')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(task + ' Loss')
    plt.ylim(0, 0.3)
    plt.show()


def readfromLOG():
    with open(file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "mmengine - INFO - train  " + task + " precision" in line:
                precision_class1 = line[60:66]
                precision_class0 = line[68:74]
                recall_class1 = line[83:89]
                recall_class0 = line[91:97]
                f_score_class1 = line[107:113]
                f_score_class0 = line[115:121]
                kappa = line[129:135]

                train_precision_class0.append(float(precision_class0))
                train_precision_class1.append(float(precision_class1))
                train_recall_class0.append(float(recall_class0))
                train_recall_class1.append(float(recall_class1))
                train_f_score_class0.append(float(f_score_class0))
                train_f_score_class1.append(float(f_score_class1))
                train_kappa.append(float(kappa))

            elif "mmengine - INFO - val    " + task + " precision" in line:
                precision_class1 = line[60:66]
                precision_class0 = line[68:74]
                recall_class1 = line[83:89]
                recall_class0 = line[91:97]
                f_score_class1 = line[107:113]
                f_score_class0 = line[115:121]
                kappa = line[129:135]

                val_precision_class0.append(float(precision_class0))
                val_precision_class1.append(float(precision_class1))
                val_recall_class0.append(float(recall_class0))
                val_recall_class1.append(float(recall_class1))
                val_f_score_class0.append(float(f_score_class0))
                val_f_score_class1.append(float(f_score_class1))
                val_kappa.append(float(kappa))

            elif "mmengine - INFO - test   " + task + " precision" in line:
                precision_class1 = line[60:66]
                precision_class0 = line[68:74]
                recall_class1 = line[83:89]
                recall_class0 = line[91:97]
                f_score_class1 = line[107:113]
                f_score_class0 = line[115:121]
                kappa = line[129:135]

                test_precision_class0.append(float(precision_class0))
                test_precision_class1.append(float(precision_class1))
                test_recall_class0.append(float(recall_class0))
                test_recall_class1.append(float(recall_class1))
                test_f_score_class0.append(float(f_score_class0))
                test_f_score_class1.append(float(f_score_class1))
                test_kappa.append(float(kappa))


def readfromLOG2():
    with open(file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "mmengine - INFO - train  " + task + " precision" in line:
                if task == 'T2':
                    precision_match = re.search(precision_pattern, line)
                    precision_class2 = float(precision_match.group(1))
                    precision_class1 = float(precision_match.group(2))
                    precision_class0 = float(precision_match.group(3))

                    recall_match = re.search(recall_pattern, line)
                    recall_class2 = float(recall_match.group(1))
                    recall_class1 = float(recall_match.group(2))
                    recall_class0 = float(recall_match.group(3))

                    fscore_match = re.search(fscore_pattern, line)
                    f_score_class2 = float(fscore_match.group(1))
                    f_score_class1 = float(fscore_match.group(2))
                    f_score_class0 = float(fscore_match.group(3))

                    kappa_match = re.search(kappa_pattern, line)
                    kappa = float(kappa_match.group(1))

                    train_precision_class2.append(precision_class2)
                    train_recall_class2.append(recall_class2)
                    train_f_score_class2.append(f_score_class2)
                else:
                    precision_match = re.search(precision_pattern, line)
                    precision_class1 = float(precision_match.group(1))
                    precision_class0 = float(precision_match.group(2))

                    recall_match = re.search(recall_pattern, line)
                    recall_class1 = float(recall_match.group(1))
                    recall_class0 = float(recall_match.group(2))

                    fscore_match = re.search(fscore_pattern, line)
                    f_score_class1 = float(fscore_match.group(1))
                    f_score_class0 = float(fscore_match.group(2))

                    kappa_match = re.search(kappa_pattern, line)
                    kappa = float(kappa_match.group(1))

                train_precision_class0.append(precision_class0)
                train_precision_class1.append(precision_class1)
                train_recall_class0.append(recall_class0)
                train_recall_class1.append(recall_class1)
                train_f_score_class0.append(f_score_class0)
                train_f_score_class1.append(f_score_class1)
                train_kappa.append(kappa)

            elif "mmengine - INFO - val    " + task + " precision" in line:
                if task == 'T2':
                    precision_match = re.search(precision_pattern, line)
                    precision_class2 = float(precision_match.group(1))
                    precision_class1 = float(precision_match.group(2))
                    precision_class0 = float(precision_match.group(3))

                    recall_match = re.search(recall_pattern, line)
                    recall_class2 = float(recall_match.group(1))
                    recall_class1 = float(recall_match.group(2))
                    recall_class0 = float(recall_match.group(3))

                    fscore_match = re.search(fscore_pattern, line)
                    f_score_class2 = float(fscore_match.group(1))
                    f_score_class1 = float(fscore_match.group(2))
                    f_score_class0 = float(fscore_match.group(3))

                    kappa_match = re.search(kappa_pattern, line)
                    kappa = float(kappa_match.group(1))

                    val_precision_class2.append(precision_class2)
                    val_recall_class2.append(recall_class2)
                    val_f_score_class2.append(f_score_class2)
                else:
                    precision_match = re.search(precision_pattern, line)
                    precision_class1 = float(precision_match.group(1))
                    precision_class0 = float(precision_match.group(2))

                    recall_match = re.search(recall_pattern, line)
                    recall_class1 = float(recall_match.group(1))
                    recall_class0 = float(recall_match.group(2))

                    fscore_match = re.search(fscore_pattern, line)
                    f_score_class1 = float(fscore_match.group(1))
                    f_score_class0 = float(fscore_match.group(2))

                    kappa_match = re.search(kappa_pattern, line)
                    kappa = float(kappa_match.group(1))

                val_precision_class0.append(precision_class0)
                val_precision_class1.append(precision_class1)
                val_recall_class0.append(recall_class0)
                val_recall_class1.append(recall_class1)
                val_f_score_class0.append(f_score_class0)
                val_f_score_class1.append(f_score_class1)
                val_kappa.append(kappa)

            elif "mmengine - INFO - test   " + task + " precision" in line:
                if task == 'T2':
                    precision_match = re.search(precision_pattern, line)
                    precision_class2 = float(precision_match.group(1))
                    precision_class1 = float(precision_match.group(2))
                    precision_class0 = float(precision_match.group(3))

                    recall_match = re.search(recall_pattern, line)
                    recall_class2 = float(recall_match.group(1))
                    recall_class1 = float(recall_match.group(2))
                    recall_class0 = float(recall_match.group(3))

                    fscore_match = re.search(fscore_pattern, line)
                    f_score_class2 = float(fscore_match.group(1))
                    f_score_class1 = float(fscore_match.group(2))
                    f_score_class0 = float(fscore_match.group(3))

                    kappa_match = re.search(kappa_pattern, line)
                    kappa = float(kappa_match.group(1))

                    test_precision_class2.append(precision_class2)
                    test_recall_class2.append(recall_class2)
                    test_f_score_class2.append(f_score_class2)
                else:
                    precision_match = re.search(precision_pattern, line)
                    precision_class1 = float(precision_match.group(1))
                    precision_class0 = float(precision_match.group(2))

                    recall_match = re.search(recall_pattern, line)
                    recall_class1 = float(recall_match.group(1))
                    recall_class0 = float(recall_match.group(2))

                    fscore_match = re.search(fscore_pattern, line)
                    f_score_class1 = float(fscore_match.group(1))
                    f_score_class0 = float(fscore_match.group(2))

                    kappa_match = re.search(kappa_pattern, line)
                    kappa = float(kappa_match.group(1))

                test_precision_class0.append(precision_class0)
                test_precision_class1.append(precision_class1)
                test_recall_class0.append(recall_class0)
                test_recall_class1.append(recall_class1)
                test_f_score_class0.append(f_score_class0)
                test_f_score_class1.append(f_score_class1)
                test_kappa.append(kappa)

            elif task + "_loss" in line:
                loss_match = re.search(loss_pattern, line)
                loss = float(loss_match.group(1))
                loss_value.append(loss)
            elif "max_epochs" in line:
                epoch_match = re.search(epoch_pattern, line)
                epoch = int(epoch_match.group(1))
                ALLepoch.append(epoch)


# INIT ALL DATA
train_precision_class0 = []
train_precision_class1 = []
train_recall_class0 = []
train_recall_class1 = []
train_f_score_class0 = []
train_f_score_class1 = []
train_kappa = []

val_precision_class0 = []
val_precision_class1 = []
val_recall_class0 = []
val_recall_class1 = []
val_f_score_class0 = []
val_f_score_class1 = []
val_kappa = []

test_precision_class0 = []
test_precision_class1 = []
test_recall_class0 = []
test_recall_class1 = []
test_f_score_class0 = []
test_f_score_class1 = []
test_kappa = []

loss_value = []
ALLepoch = []

# FILE NAME:
file_name = "20230626_230823"
task = 'T1'
file = os.path.join("/home/chenqiongpu/SLO/SLO/work_dirs/SLO", file_name,
                          file_name + '.log')


if task == 'T2':
    precision_pattern = r"precision\[\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    recall_pattern = r"recall\[\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    fscore_pattern = r"f-score\[\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    kappa_pattern = r"kappa\[\s*([-]?\d+\.\d+)\]"
    loss_pattern = task + r"_loss:\s*(\d+\.\d+)"

    train_precision_class2 = []
    train_recall_class2 = []
    train_f_score_class2 = []
    val_precision_class2 = []
    val_recall_class2 = []
    val_f_score_class2 = []
    test_precision_class2 = []
    test_recall_class2 = []
    test_f_score_class2 = []
else:
    precision_pattern = r"precision\[\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    recall_pattern = r"recall\[\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    fscore_pattern = r"f-score\[\s*(\d+\.\d+),\s*(\d+\.\d+)\]"
    kappa_pattern = r"kappa\[\s*([-]?\d+\.\d+)\]"
    loss_pattern = task + r"_loss:\s*(\d+\.\d+)"
    epoch_pattern = r"max_epochs=(\d{1,3})"

if __name__ == '__main__':
    # read from log
    # readfromLOG()
    readfromLOG2()

    # DISPLAY 4 metrics
    displayPrecisionMetrics()
    displayRecallMetrics()
    displayFscoreMetrics()
    displayKappaMetrics()
    drawLoss()
