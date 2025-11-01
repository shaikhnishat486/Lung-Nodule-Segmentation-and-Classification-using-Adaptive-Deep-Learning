import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.patheffects as pe
from sklearn import metrics

No_of_Dataset = 2


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'SSA-ARMVAM', 'DBO-ARMVAM', 'AOA-ARMVAM', 'GEA-ARMVAM', 'RVO-GEA-ARMVAM']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        fig = plt.figure(facecolor='#f6f6f6')
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
        fig.canvas.manager.set_window_title('Dataset - ' + str(i + 1) + ' - Convergence Curve')
        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='SSA-ARMVAM')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='DBO-ARMVAM')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='AOA-ARMVAM')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='GEA-ARMVAM')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='RVO-GEA-ARMVAM')
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=12, fontweight='bold', color='k')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    cls = ['Dilated RAN', 'MobileNetV2', 'VGG16', 'MViT-AM', 'ARMVAM']
    for a in range(No_of_Dataset):  # For 2 Datasets
        Actual = np.load('Target_1.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        Actual = Actual[lenper:, :]
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset - ' + str(a + 1) + ' - ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_%s.png" % (a + 1)
        plt.savefig(path)
        plt.show()


def Table():
    eval = np.load('Evaluates_Epoch.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'SSA-ARMVAM', 'DBO-ARMVAM', 'AOA-ARMVAM', 'GEA-ARMVAM', 'RVO-GEA-ARMVAM']
    Classifier = ['BatchSize', 'Dilated RAN', 'MobileNetV2', 'VGG16', 'MViT-AM', 'ARMVAM']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Term = np.array([0, 2, 9, 18]).astype(int)
    Table_Terms = [0, 2, 9, 18]
    table_terms = [Terms[i] for i in Table_Terms]
    Batch_size = [4, 16, 32, 64, 128]
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batch_size)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batch_size)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plot_Results():
    eval = np.load('Evaluates_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 5, 8, 10, 12]
    Kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(facecolor='#FFE4E1')
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Algorithm Comparison of Kfold')
            plt.plot(Kfold, Graph[:, 0], lw=5, color='blue',
                     path_effects=[pe.withStroke(linewidth=8, foreground='violet')], marker='h',
                     markerfacecolor='blue', markersize=5,
                     label="SSA-ARMVAM")
            plt.plot(Kfold, Graph[:, 1], lw=5, color='maroon',
                     path_effects=[pe.withStroke(linewidth=8, foreground='tan')], marker='h',
                     markerfacecolor='#7FFF00', markersize=5,
                     label="DBO-ARMVAM")
            plt.plot(Kfold, Graph[:, 2], lw=5, color='lime',
                     path_effects=[pe.withStroke(linewidth=8, foreground='orange')], marker='h',
                     markerfacecolor='#808000',
                     markersize=5,
                     label="AOA-ARMVAM")
            plt.plot(Kfold, Graph[:, 3], lw=5, color='deeppink',
                     path_effects=[pe.withStroke(linewidth=8, foreground='w')], marker='h', markerfacecolor='#CD1076',
                     markersize=5,
                     label="GEA-ARMVAM")
            plt.plot(Kfold, Graph[:, 4], lw=5, color='k',
                     path_effects=[pe.withStroke(linewidth=8, foreground='red')], marker='h', markerfacecolor='black',
                     markersize=5,
                     label="RVO-GEA-ARMVAM")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(Kfold, ('1', '2', '3', '4', '5'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.xlabel('Kfold', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure(facecolor='#FFE4E1')
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Method Comparison of Kfold')
            X = np.arange(len(Kfold))
            ax.bar(X + 0.00, Graph[:, 5], color='orangered', edgecolor='cyan', width=0.15, label="Dilated RAN")
            ax.bar(X + 0.15, Graph[:, 6], color='fuchsia', edgecolor='cyan', width=0.15, label="MobileNetV2")
            ax.bar(X + 0.30, Graph[:, 7], color='indigo', edgecolor='cyan', width=0.15, label="VGG16")
            ax.bar(X + 0.45, Graph[:, 8], color='brown', edgecolor='cyan', width=0.15, label="MViT-AM")
            ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='cyan', width=0.15, label="ARMVAM")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.30, ('1', '2', '3', '4', '5'), fontname="Arial", fontsize=12, fontweight='bold',
                       color='#35530a')
            plt.xlabel('Kfold', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            path = "./Results/Dataset_%s_%s_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Confusion():
    no_of_Dataset = 2
    for n in range(no_of_Dataset):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True)
        class1 = ['Normal', 'Nodule']
        class2 = ['Normal', 'Nodule']
        classes = [class1, class2]
        if n == 0:
            confusion_matrix = metrics.confusion_matrix(Actual, Predict)
        else:
            confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes[n])
        cm_display.plot()
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title("Confusion Matrix")
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        # plt.savefig(path)
        plt.show()


def plot_seg_results():
    Eval_all = np.load('Seg_Eval.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Methods = ['TERMS', 'Trans-MobileUnet', 'Trans-DenseUNet', 'Trans-ResUnet', 'Trans-ResUnet++', 'TD-DASPP']
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            Table = PrettyTable()
            Table.add_column(Methods[0], Statistics[::4])
            Table.add_column(Methods[1], stats[i, 5, ::4])
            Table.add_column(Methods[2], stats[i, 6, ::4])
            Table.add_column(Methods[3], stats[i, 7, ::4])
            Table.add_column(Methods[4], stats[i, 8, ::4])
            Table.add_column(Methods[5], stats[i, 4, ::4])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Comparison for Segmentation of dataset', n + 1, '--------------------------------------------------')
            print(Table)

            X = np.arange(len(Statistics) - 4)
            name = [0, 0.2, 0.4, 0.6, 0.8]
            fig = plt.figure(facecolor='#F4ACB7')
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset-' + str(n+1) + '- Mean- ' + str(Terms[i - 4] + 'Method Comparison'))
            ax.bar(X + 0.00, stats[i, 0, 2], color='#8A2BE2', edgecolor='k', width=0.10, label="Trans-MobileUNet")
            ax.bar(X + 0.20, stats[i, 1, 2], color='#DC143C', edgecolor='k', width=0.10, label="Trans-DenseUNet")
            ax.bar(X + 0.40, stats[i, 2, 2], color='#FF00FF', edgecolor='k', width=0.10, label="Trans-ResUNet")
            ax.bar(X + 0.60, stats[i, 3, 2], color='lime', edgecolor='k', width=0.10, label="Trans-ResUnet++")
            ax.bar(X + 0.80, stats[i, 4, 2], color='k', edgecolor='cyan', width=0.10, label="TD-DASPP")
            plt.xticks(name, ('Trans-MobileUnet', 'Trans-DenseUnet', 'Trans-ResUnet', 'Trans-ResUnet++', 'TD-DASPP'), rotation=12)
            plt.ylabel(Terms[i - 4])
            path = "./Results/Dataset_%s_%s_Mean_met.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path)
            plt.show()


def Exiting_Table():
    eval = np.load('Eval.npy', allow_pickle=True)
    eval_1 = np.load('Evaluates_Epoch.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'SSA', 'DBO', 'AOA', 'GEA', 'Proposed']
    Classifier = ['BatchSize', 'Mod 1', 'Mod 2', 'Mod 3', 'Mod 4', 'Proposed']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Term = np.array([0, 2, 9, 18]).astype(int)
    Table_Terms = [0, 2, 9, 18]
    table_terms = [Terms[i] for i in Table_Terms]
    Batch_size = [4, 16, 32, 64, 128]
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]
            value_1 = eval_1[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batch_size)
            for j in range(len(Classifier) - 2):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term[k]])
            Table.add_column(Classifier[5], value_1[:, 4, Graph_Term[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Existing  Comparison',
                  '---------------------------------------')
            print(Table)


if __name__ == '__main__':
    plotConvResults()
    Plot_Results()
    plot_seg_results()
    Plot_ROC_Curve()
    Table()
    Exiting_Table()
