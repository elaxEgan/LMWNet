import numpy as np




precisions = []
recalls = []
precision_list = []
recall_list = []
precision_total = []
recall_total = []
mae_total = []

thresholds = np.arange(256) / 255  # 阈值从0到255


# thresholds = [0.5]


def calculate_precision_recall(gt, res):
    for threshold in thresholds:
        # 将预测结果二值化
        binary_res = (res > threshold).astype(int)
        binary_gt = (gt > threshold).astype(int)

        # 计算准确率和召回率
        TP = np.sum((binary_gt == 1) & (binary_res == 1))
        FP = np.sum((binary_gt == 0) & (binary_res == 1))
        FN = np.sum((binary_gt == 1) & (binary_res == 0))
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def calculate_fbeta(precisions, recalls, beta):
    fbeta_list = []
    for precision, recall in zip(precisions, recalls):
        if precision + recall != 0:
            fbeta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
            fbeta_list.append(fbeta)
        else:
            fbeta_list.append(0.0)
    return fbeta_list


def calculate_E(res, gt):
    H, W = gt.shape
    E = 0
    for x in range(H):
        for y in range(W):
            E += 1.0 / (1 + np.exp(-(np.abs(res[x, y] - gt[x, y]))))
    E /= H * W
    return E


def calculate_mae(S, GT):
    H, W = S.shape
    mae = (1 / (W * H)) * np.sum(np.abs(S - GT))
    return mae


