from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image

plt.rcParams['font.size'] = 14
InteractiveShell.ast_node_interactivity = 'all'


class InputDataAnalyser:
    def __init__(self, train_dir, valid_dir, test_dir):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir

        self.prepare_stats()

    def prepare_stats(self):
        # Empty lists
        categories = []
        img_categories = []
        n_train = []
        n_valid = []
        n_test = []
        hs = []
        ws = []

        # Iterate through each category
        for d in os.listdir(self.train_dir):
            categories.append(d)

            # Number of each image
            train_imgs = os.listdir(self.train_dir + d)
            valid_imgs = os.listdir(self.valid_dir + d)
            test_imgs = os.listdir(self.test_dir + d)
            n_train.append(len(train_imgs))
            n_valid.append(len(valid_imgs))
            n_test.append(len(test_imgs))

            # Find stats for train images
            for i in train_imgs:
                img_categories.append(d)
                img = Image.open(self.train_dir + d + '/' + i)
                img_array = np.array(img)
                # Shape
                hs.append(img_array.shape[0])
                ws.append(img_array.shape[1])

        # Dataframe of categories
        cat_df = pd.DataFrame({'category': categories,
                               'n_train': n_train,
                               'n_valid': n_valid, 'n_test': n_test}). \
            sort_values('category')

        # Dataframe of training images
        image_df = pd.DataFrame({
            'category': img_categories,
            'height': hs,
            'width': ws
        })

        cat_df.sort_values('n_train', ascending=False, inplace=True)
        cat_df.head()
        cat_df.tail()

        cat_df.set_index('category')['n_train'].plot.bar(color='r', figsize=(20, 6))

        display(cat_df)

        plt.xticks(rotation=80)
        plt.ylabel('Count')
        plt.title('Training Images by Category')
        plt.show()
