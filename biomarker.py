from opt import *
import csv
import codecs


def csv_save(file_name, datas):   # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

opt = OptInit().initialize()
path = opt.ckpt_path

n_exp = opt.num_exp
n_fold = opt.num_fold

imp_ROI_name_list = []
imp_ROI_idx_list = []

for exp in range(n_exp):
    print("========== EXP {} ==========".format(exp))
    ckpt_path = opt.ckpt_path + "/exp{}".format(exp)

    exp_imp_ROI_name_list = []
    exp_imp_ROI_idx_list = []

    ROI_support_path = ckpt_path + "/fold0_support_MRI.npy"
    ROI_support = np.load(ROI_support_path)

    ROI_name = []
    with open("ROI_name.csv") as f:
        f_csv = csv.reader(f)
        for i, name in enumerate(f_csv):
            if ROI_support[i]:
                ROI_name.append(name)

    for fold in range(n_fold):

        # print("========== FOLD {} ==========".format(fold))
        ROI_att_path = ckpt_path + "/fold{}_ROI.npy".format(fold)
        ROI_att = np.load(ROI_att_path)

        ROI_att = abs(ROI_att)
        imp_ROI_att = sorted(ROI_att)[-5:]

        imp_ROI_idx = [i for i, x in enumerate(ROI_att) if x in imp_ROI_att]

        imp_ROI_name = [ROI_name[i] for i in imp_ROI_idx]

        exp_imp_ROI_name_list.append(imp_ROI_name)
        exp_imp_ROI_idx_list.append(imp_ROI_idx)

        imp_ROI_name_list.append(imp_ROI_name)
        imp_ROI_idx_list.append(imp_ROI_idx)

    csv_save("impROI/exp{}/{}_imp_ROI_Name.csv".format(exp, opt.task), exp_imp_ROI_name_list)

dic = {}
for i in range(len(imp_ROI_idx_list)):
    for idx in imp_ROI_idx_list[i]:
        v = dic.get(idx)
        if v is not None:
            dic[idx] = v + 1
        else:
            dic[idx] = 1
print(dic)

dic = sorted(dic.items(), key=lambda x: x[1])
print(dic)