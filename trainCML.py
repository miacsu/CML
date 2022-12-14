'''

multi-stage multi-modal learning method based cross-modal feature fusion (SNP and MRI)
and multi-graph fusion (non-meta data and meta data)

'''

from models.multistageMultimodalLearning import CML
from opt import *
from utils import *
from metrics import metrics


def main(opt, exp, seed):

    data_folder = "./data/{}/".format(opt.task)

    # load data
    print('Loading dataset ...')
    features1 = np.loadtxt(os.path.join(data_folder, "MRI.csv"), delimiter=',')
    features2 = np.loadtxt(os.path.join(data_folder, "SNP.csv"), delimiter=',')
    features3 = np.loadtxt(os.path.join(data_folder, "meta.csv"), delimiter=',')
    y = np.loadtxt(os.path.join(data_folder, "labels.csv"), delimiter=',')

    # split data
    n_folds = opt.num_fold
    cv_splits = data_split(features1, y, n_folds, seed)

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    sens = np.zeros(n_folds, dtype=np.float32)
    spes = np.zeros(n_folds, dtype=np.float32)

    for fold in range(n_folds):
        print("========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]
        val_ind = cv_splits[fold][1]

        print('\tPreprocessing data...')
        # preprocess
        features1, support1 = get_node_features(features1, y, train_ind, 300, True)
        features2, support2 = get_node_features(features2, y, train_ind, 300, True)

        # build network architecture
        model = CML(features1.shape[1], features2.shape[1], 2*features3.shape[1], 300, 300, opt.num_classes,
                    opt.dropout, opt.hgc, opt.lg).to(opt.device)
        model = model.to(opt.device)

        # build loss, optimizer, metric
        weight = np.array([len(y[train_ind][np.where(y[train_ind] != 0)])/len(train_ind),
                           len(y[train_ind][np.where(y[train_ind] != 1)])/len(train_ind)])
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weight).float())
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)

        x1 = torch.tensor(features1, dtype=torch.float32).to(opt.device)
        x2 = torch.tensor(features2, dtype=torch.float32).to(opt.device)
        x3 = torch.tensor(features3, dtype=torch.float32).to(opt.device)
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)

        if not os.path.exists(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)
        ckpt_path = opt.ckpt_path + "/exp{}".format(exp)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        MRI_support_path = ckpt_path + "/fold{}_support_MRI".format(fold)
        np.save(MRI_support_path, support1)

        SNP_support_path = ckpt_path + "/fold{}_support_SNP".format(fold)
        np.save(SNP_support_path, support2)

        fold_model_path = ckpt_path + "/fold{}.pth".format(fold)
        fold_loss_path = ckpt_path + "/fold{}_loss".format(fold)
        fold_ROI_path = ckpt_path + "/fold{}_ROI".format(fold)
        fold_SNP_path = ckpt_path + "/fold{}_SNP".format(fold)

        def train():
            print("Start training...")
            print("Number of training samples %d" % len(train_ind))

            acc = 0
            loss_list = []
            for epoch in range(opt.num_iter):
                model.train()
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    node_logits, fusion_feat, _, _, _, _ = model(x1, x2, x3)
                    loss = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss.backward()
                    optimizer.step()
                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                model.eval()
                with torch.set_grad_enabled(False):
                    node_logits, fusion_feat, att_MRI, att_SNP, mlp1_weight, mlp2_weight = model(x1, x2, x3)
                logits_val = node_logits[val_ind].detach().cpu().numpy()
                correct_val, acc_val = accuracy(logits_val, y[val_ind])

                loss_list.append(loss.item())

                print("Epoch: {},\ttrain loss: {:.4f},\ttrain acc: {:.4f}, \tval acc: {:.4f}".format(
                    epoch, loss.item(), acc_train.item(), acc_val.item()))

                if acc_val >= acc:
                    acc = acc_val
                    torch.save(model.state_dict(), fold_model_path)
                    impROI = torch.matmul(torch.sum(att_MRI, 0), mlp1_weight)
                    impSNP = torch.matmul(torch.sum(att_SNP, 0), mlp2_weight)
                    np.save(fold_ROI_path, impROI.detach().numpy())
                    np.save(fold_SNP_path, impSNP.detach().numpy())

            accs[fold] = acc
            loss_np = np.array(loss_list)
            np.save(fold_loss_path, loss_np)
            print("\tFold {} val accuacry {:.2f}%".format(fold, acc * 100))

        def test(stage="test"):
            print('Start testing...')
            print("Number of testing samples %d" % len(test_ind))

            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            node_logits, _, _, _, _, _ = model(x1, x2, x3)

            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = get_auc(logits_test, y[test_ind])

            score_file_path = ckpt_path + '/fold{}_score_{}.txt'.format(fold, stage)
            f = open(score_file_path, 'w')
            write_raw_score(f, logits_test, y[test_ind])
            matrix = [[0, 0], [0, 0]]
            matrix = matrix_sum(matrix, get_confusion_matrix(logits_test, y[test_ind]))
            sens[fold] = get_sen(matrix)
            spes[fold] = get_spe(matrix)
            f.close()

        train()
        test()


if __name__ == '__main__':
    opt = OptInit().initialize()
    seeds = np.random.randint(0, 1000, opt.num_exp)
    print(seeds)
    for exp in range(opt.num_exp):
        main(opt, exp, seeds[exp])
    metrics(opt)