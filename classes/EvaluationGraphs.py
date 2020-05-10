import matplotlib.pyplot as plt


class EvaluationGraphs:
    @staticmethod
    def plot_roc_curve(model, dataloader):
        plt.figure()
        n_classes = ['dot', 'overlap']
        """
        compute ROC curve and ROC area for each class in each fold

        """

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(n_classes)):
            actuals, probabilities = test_class_probabilities(model, dataloader, i)
            fpr[i], tpr[i], threshold = roc_curve(actuals, probabilities)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # plt.figure(figsize=(6,6))
        for i in range(len(n_classes)):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))  # roc_auc_score

        plt.plot([0, 1], [0, 1], 'k--')
        # plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        # plt.tight_layout()
        plt.show()