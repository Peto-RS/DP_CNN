from final.GlobalSettings import GlobalSettings

import numpy as np
import os
import torch

import pandas as pd


class ModelEvaluation:
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
    def evaluate(model, test_loader, criterion, topk=(1, 5)):
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

        # Send results to a dataframe and calculate average across classes
        results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
        results['class'] = classes
        results['loss'] = losses
        results = results.groupby(classes).mean()

        return results.reset_index().rename(columns={'index': 'class'})