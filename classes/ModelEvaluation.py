from builtins import staticmethod
from datetime import datetime

from sklearn.metrics import roc_curve, auc, accuracy_score
from torch import nn
from PIL import Image

from classes.Utils import Utils
from old.GlobalSettings import GlobalSettings

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as patches

import numpy as np
import os
import seaborn as sns
import torch

import pandas as pd


class ModelEvaluation:
    @staticmethod
    def train_valid_acc_loss_graph(history, model_name, save, training_evaluation_directory):
        history = pd.DataFrame(history, columns=['train_acc', 'train_loss', 'valid_acc', 'valid_loss'])

        fig = plt.figure(figsize=(15, 6))
        locator = matplotlib.ticker.MultipleLocator(1)
        fig.suptitle(model_name)

        subfig = fig.add_subplot(122)
        fig.gca().xaxis.set_major_locator(locator)
        subfig.plot(history['train_acc'], label="Training")
        subfig.plot(history['valid_acc'], label="Validation")
        subfig.set_title('Model Accuracy')
        subfig.set_xlabel('Epoch')
        subfig.set_ylabel('Percentage')
        subfig.legend(loc='upper left')

        subfig = fig.add_subplot(121)
        fig.gca().xaxis.set_major_locator(locator)
        subfig.plot(history['train_loss'], label="Training")
        subfig.plot(history['valid_loss'], label="Validation")
        subfig.set_title('Model Loss')
        subfig.set_xlabel('Epoch')
        subfig.set_ylabel('Loss')
        subfig.legend(loc='upper right')

        plt.show()

        if save:
            path = os.path.join(training_evaluation_directory, model_name)
            path = Utils.create_folder_if_not_exists(path)

            plt.savefig(os.path.join(path, 'valid_acc_loss_graph_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.png'),
                        bbox_inches='tight')

    @staticmethod
    def plot_roc_curve(model, dataloader, classes):
        plt.figure()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(classes)):
            actuals, probabilities = test_class_probabilities(model, dataloader, i)
            fpr[i], tpr[i], threshold = roc_curve(actuals, probabilities)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # plt.figure(figsize=(6,6))
        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(classes[i], roc_auc[i]))  # roc_auc_score

        # # # Compute micro-average ROC curve and ROC area
        # all_labels = sum([i for i in range(len(classes))] * len(p))
        # fpr["micro"], tpr["micro"], _ = roc_curve(all_labels, sum(p, []))
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.plot([0, 1], [0, 1], 'k--')
        # plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.tight_layout()
        plt.show()

    @staticmethod
    def get_predictions(model, dataloader):
        model.to(torch.device("cuda:0"))
        model.eval()
        labels = []
        predictions = []
        probabilities = []
        for i, batch in enumerate(dataloader):
            data, label = batch  # ignore label
            data = data.cuda()
            label = label.cuda()
            outputs = model(data)
            _, preds = torch.max(outputs.data, 1)
            labels.extend(label.tolist())
            predictions.extend(preds.tolist())
            m = torch.nn.Softmax(dim=1)
            outputs = m(outputs)
            probabilities.extend(outputs.tolist())

        return labels, predictions, probabilities

    @staticmethod
    def predict_fingerprint(pil_image, model, class_names, all_classes, sliding_window_height, sliding_window_width, step,
                probability_threshold, tensor_height, tensor_width):
        model.eval()
        model.to('cuda:0')

        colors = {'bifurcation': 'lime', 'ending': 'r', 'nothing': 'y', 'dot': 'm', 'overlap': 'b'}

        image_width = tensor_width
        image_height = tensor_height

        def pr(model, img_tensor):
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)

                m = torch.nn.Softmax(dim=1)
                outputs = m(outputs)
                # print(outputs)
                return predicted, outputs[0][predicted]

        image = pil_image

        fig, ax = plt.subplots(1)

        stepSize = step
        w_width, w_height = sliding_window_width, sliding_window_height  # window size
        for x in range(0, image.shape[1] - w_width, stepSize):
            for y in range(0, image.shape[0] - w_height, stepSize):
                window = image[x:x + w_width, y:y + w_height, :]
                # plt.imshow(window)
                # plt.show()
                img_tensor = process_image(window, image_width, image_height)
                # plt.imshow(img_tensor.permute(1, 2, 0))
                # plt.show()
                img_tensor = img_tensor.view(1, 3, image_height, image_width).cuda()

                pred, prob = pr(model, img_tensor)
                predicted_class_string = all_classes[pred]
                if predicted_class_string in class_names and prob > probability_threshold:
                    color = colors[predicted_class_string] if (predicted_class_string in colors) else 'k'
                    ax.add_patch(
                        patches.Rectangle((x, y), w_width, w_height, edgecolor=color, fill=False))

        legend_elements = []
        for class_name in all_classes:
            color = colors[class_name] if (class_name in colors) else 'k'

            legend_elements.append(patches.Rectangle((0, 0), 20, 20, color=color, label=class_name))

        ax.legend(handles=legend_elements, loc='upper right')
        ax.imshow(pil_image)
        plt.show()

    @staticmethod
    def predict(pil_image, model, all_classes, tensor_height, tensor_width):
        model.eval()
        model.to('cuda:0')

        colors = {'bifurcation': 'lime', 'ending': 'r', 'nothing': 'y', 'dot': 'm', 'overlap': 'b'}

        image_width = tensor_width
        image_height = tensor_height

        def pr(model, img_tensor):
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)

                m = torch.nn.Softmax(dim=1)
                outputs = m(outputs)
                # print(outputs)
                return predicted, outputs[0][predicted]

        image = pil_image

        # fig, ax = plt.subplots(1)

        img_tensor = process_image(image, image_width, image_height)
        img_tensor = img_tensor.view(1, 3, image_height, image_width).cuda()

        pred, prob = pr(model, img_tensor)
        return all_classes[pred]

    @staticmethod
    def get_accuracy(dataset_dataloader_test, model):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataset_dataloader_test:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    @staticmethod
    def get_accuracy_classes(dataset_dataloader_test, model):
        classes = dataset_dataloader_test.dataset.classes

        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            for data in dataset_dataloader_test:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(classes)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        accuracy_classes = ''
        for i in range(len(classes)):
            accuracy_classes += 'Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i])


def test_class_probabilities(model, test_loader, n_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, labels = batch

            outputs = model(inputs).squeeze()

            prediction = outputs.argmax(dim=1, keepdim=True)
            updated_list = [int(bool) for bool in (labels == n_class).tolist()]
            actuals.extend(updated_list)

            m = torch.nn.Softmax(dim=1)
            outputs = m(outputs)
            probabilities.extend(outputs[:, n_class].tolist())

    return actuals, probabilities  # [i.item() for i in actuals], [i.item() for i in probabilities]


@staticmethod
def get_top1_accuracy(results, save_on_disk):
    sns.lmplot(y='top1', x='n_train', data=results, height=6)
    plt.xlabel('Image Count')
    plt.ylabel('Accuracy (%)')
    plt.title(str(GlobalSettings.CNN_MODEL.value[0]).upper() + '\nTop 1 Accuracy vs Number of Training Images')
    plt.ylim(-5, 105)

    if save_on_disk:
        plt.savefig(GlobalSettings.SAVE_FOLDER + '/' + str(GlobalSettings.CNN_MODEL.value[0]) + '/top1.png',
                    bbox_inches='tight')
    else:
        plt.show()


def process_image(pil_image, img_width, img_height):
    # Resize
    img = Image.fromarray(pil_image).resize((img_height, img_width))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


@staticmethod
def accuracy(output, target, topk=(1,)):
    """Compute the topk accuracy(s)"""
    output = output.to('cuda')
    target = target.to('cuda')

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


@staticmethod
def evaluate(model, test_loader, criterion, topk=(1, GlobalSettings.NUM_CLASSES)):
    """Measure the performance of a trained PyTorch app_model

    Params
    --------
        app_model (PyTorch app_model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros((len(test_loader.dataset), len(topk)))
    i = 0

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:
            data, targets = data.to('cuda'), targets.to('cuda')

            # Raw app_model output
            out = model(data)
            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk accuracy
                acc_results[i, :] = ModelEvaluation.accuracy(pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(model.idx_to_class[true.item()])
                # Calculate the loss
                loss = criterion(pred.view(1, len(os.listdir(GlobalSettings.TRAIN_DIR))), true.view(1))
                losses.append(loss.item())
                i += 1

    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results['class'] = classes
    results['loss'] = losses
    results = results.groupby(classes).mean()

    return results.reset_index().rename(columns={'index': 'class'})


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataset_dataloader_test, num_images=9):
    classes = dataset_dataloader_test.dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataset_dataloader_test):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('{}'.format(classes[preds[j]]))
                ModelEvaluation.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        plt.show()


@staticmethod
def confusion_matrix(self):
    nb_samples = 20
    nb_classes = 4
    output = torch.randn(nb_samples, nb_classes)
    pred = torch.argmax(output, 1)
    target = torch.randint(0, nb_classes, (nb_samples,))

    conf_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(target, pred):
        conf_matrix[t, p] += 1

    print('Confusion matrix\n', conf_matrix)

    TP = conf_matrix.diag()
    for c in range(nb_classes):
        idx = torch.ones(nb_classes).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN = conf_matrix[idx.nonzero()[:,
                         None], idx.nonzero()].sum()  # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN = conf_matrix[c, idx].sum()

        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))


@staticmethod
def confusion_matrix_image():
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import plot_confusion_matrix

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
