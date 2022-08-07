import matplotlib.pyplot as plt
from utils import *


def metrics(opt):
    path = opt.ckpt_path
    accs = []
    sens = []
    spes = []
    aucs = []
    for exp in range(opt.num_exp):
        n_fold = opt.num_fold
        matrix = [[0, 0], [0, 0]]
        preds = []
        labels = []
        for i in range(n_fold):

            with open(path + '/exp{}/fold{}_score_test.txt'.format(exp, i), 'r') as f:
                file = f.readlines()
                for j in range(len(file)):
                    data = file[j].split("__")
                    tmp = [data[0], data[1]]
                    tmp = np.array(tmp).astype(np.float32)
                    preds.append(tmp)
                    label = data[2].split()[0]
                    labels.append(round(float(label)))
            f.close()
        matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
        acc = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
        sen = matrix[1][1] / (matrix[0][1] + matrix[1][1])
        spe = matrix[0][0] / (matrix[0][0] + matrix[1][0])
        auc = get_auc(preds, labels)
        print(matrix)

        print("acc: {:.4}%".format(acc*100))

        print("sen: {:.4}%".format(sen*100))

        print("spe: {:.4}%".format(spe*100))

        print("auc: {:.4}".format(auc))

        plt.title('Results of HC vs. MCI classification')

        plt.bar(0, acc, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='ACC')
        plt.bar(1.5, sen, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='SEN')
        plt.bar(3, spe, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='SPE')

        plt.xticks([0, 1.5, 3], ['ACC', 'SEN', 'SPE'])
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

        plt.savefig(path + "/exp{}/res.png".format(exp))

        accs.append(acc)
        sens.append(sen)
        spes.append(spe)
        aucs.append(auc)

    print("=> Mean of test acc : {:.2f} and Std of test acc : {:.2f}".format(np.mean(accs) * 100, np.std(accs) * 100))
    print("=> Mean of test sen : {:.2f} and Std of test sen : {:.2f}".format(np.mean(sens) * 100, np.std(sens) * 100))
    print("=> Mean of test spe : {:.2f} and Std of test spe : {:.2f}".format(np.mean(spes) * 100, np.std(spes) * 100))
    print("=> Mean of test auc : {:.4f} and Std of test auc : {:.4f}".format(np.mean(aucs), np.std(aucs)))
