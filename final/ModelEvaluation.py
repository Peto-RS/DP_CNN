from builtins import staticmethod
from final.GlobalSettings import GlobalSettings

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch

import pandas as pd


class ModelEvaluation:
    @staticmethod
    def get_top1_accuracy(results, save_on_disk):
        sns.lmplot(y='top1', x='n_train', data=results, height=6)
        plt.xlabel('Image Count')
        plt.ylabel('Accuracy (%)')
        plt.title(str(GlobalSettings.CNN_MODEL.value[0]).upper() + '\nTop 1 Accuracy vs Number of Training Images')
        plt.ylim(-5, 105)

        if save_on_disk:
            plt.savefig(GlobalSettings.SAVE_FOLDER + '/' + str(GlobalSettings.CNN_MODEL.value[0]) + '/top1.png', bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def train_valid_graph(history, save_on_disk):
        history = pd.DataFrame(history, columns=['train_acc', 'train_loss', 'valid_acc', 'valid_loss'])

        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'valid_loss']:
            plt.plot(history[c], label=("Train Loss" if c == 'train_loss' else "Valid Loss"))
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Negative Log Likelihood')
        plt.title(str(GlobalSettings.CNN_MODEL.value[0]).upper() + '\nTraining and Validation Losses')

        if save_on_disk:
            plt.savefig(GlobalSettings.SAVE_FOLDER + '/' + str(GlobalSettings.CNN_MODEL.value[0]) + '/trainingLoss.png', bbox_inches='tight')
        else:
            plt.show()

        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'valid_acc']:
            plt.plot(100 * history[c], label=("Train Acc" if c == 'train_acc' else "Valid Acc"))
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Accuracy')
        plt.title(str(GlobalSettings.CNN_MODEL.value[0]).upper() + '\nTraining and Validation Accuracy')

        if save_on_disk:
            plt.savefig(GlobalSettings.SAVE_FOLDER + '/' + str(GlobalSettings.CNN_MODEL.value[0]) + '/trainingAcc.png', bbox_inches='tight')
        else:
            plt.show()

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
        """Measure the performance of a trained PyTorch model

        Params
        --------
            model (PyTorch model): trained cnn for inference
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

                # Raw model output
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

    def visualize_model(model, testdata, class_names, num_images=30):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testdata):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    ModelEvaluation.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
