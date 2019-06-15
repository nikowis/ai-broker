from sklearn.metrics import roc_curve, auc

from stock_constants import MICRO_ROC_KEY


def calculate_roc_auc(y_test, y_test_score, classes_count):
    if classes_count > 2:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(classes_count):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY], _ = roc_curve(y_test.ravel(), y_test_score.ravel())
        roc_auc[MICRO_ROC_KEY] = auc(fpr[MICRO_ROC_KEY], tpr[MICRO_ROC_KEY])

        return fpr, tpr, roc_auc
    else:
        y_test_score = y_test_score.flatten()
        fpr, tpr, _ = roc_curve(y_test, y_test_score)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
