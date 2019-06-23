from sklearn.metrics import roc_curve, auc

from stock_constants import MICRO_ROC_KEY


def calculate_roc_auc(y_test, y_test_score, classes_count, one_hot_encode_labels):
    if classes_count > 2:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if one_hot_encode_labels:
            for i in range(classes_count):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY], _ = roc_curve(y_test.ravel(), y_test_score.ravel())
            roc_auc[MICRO_ROC_KEY] = auc(fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY])
        else:
            fpr = {0: [], 1: [], 2: [], 'micro': []}
            tpr = {0: [], 1: [], 2: [], 'micro': []}
            roc_auc = {0: 1, 1: 1, 2: 1, 'micro': 1}
        return fpr, tpr, roc_auc
    else:
        if one_hot_encode_labels:
            y_test_score = y_test_score.flatten()
            fpr, tpr, _ = roc_curve(y_test, y_test_score)
            roc_auc = auc(fpr, tpr)
        else:
            fpr = []
            tpr = []
            roc_auc = 1
        return fpr, tpr, roc_auc
