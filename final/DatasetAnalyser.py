from final.GlobalSettings import GlobalSettings
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


class DatasetAnalyser:
    @staticmethod
    def show_dataset_analysis(save_on_disk):
        categories = []
        img_categories = []
        n_train = []
        n_valid = []
        n_test = []
        hs = []
        ws = []

        for d in os.listdir(GlobalSettings.TRAIN_DIR):
            categories.append(d)

            train_imgs = os.listdir(GlobalSettings.TRAIN_DIR + '/' + d)
            valid_imgs = os.listdir(GlobalSettings.VALID_DIR + '/' + d)
            test_imgs = os.listdir(GlobalSettings.TEST_DIR + '/' + d)
            n_train.append(len(train_imgs))
            n_valid.append(len(valid_imgs))
            n_test.append(len(test_imgs))

            # Find stats for train images
            for i in train_imgs:
                img_categories.append(d)
                img = Image.open(GlobalSettings.TRAIN_DIR + '/' + d + '/' + i)
                img_array = np.array(img)

                hs.append(img_array.shape[0])
                ws.append(img_array.shape[1])

        cat_df = pd.DataFrame({
            'Category': categories,
            'n_train': n_train,
            'n_valid': n_valid,
            'n_test': n_test}
        ).sort_values('Category')

        cat_df.sort_values('n_train', ascending=False, inplace=True)
        cat_df.set_index('Category')['n_train'].plot.bar(color='r', figsize=(20, 6))

        #####
        # TRAINING IMAGES BY CATEGORY
        #####
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.title('Training Images by Category')

        if save_on_disk:
            plt.savefig(GlobalSettings.SAVE_FOLDER + '/' + 'IMAGE_COUNT_BY_CATEGORY.png', bbox_inches='tight')
        else:
            plt.show()

        image_df = pd.DataFrame({
            'Category': img_categories,
            'height': hs,
            'width': ws
        })

        img_dsc = image_df.groupby('Category').describe()
        img_dsc.head()

        #####
        # AVERAGE SIZE DISTRIBUTION
        #####
        plt.figure(figsize=(10, 6))
        sns.kdeplot(img_dsc['height']['mean'], label='Average Height')
        sns.kdeplot(
            img_dsc['width']['mean'], label='Average Width')
        plt.xlabel('Pixels')
        plt.ylabel('Density')
        plt.title('Average Size Distribution')

        if save_on_disk:
            plt.savefig(GlobalSettings.SAVE_FOLDER + '/' + 'AVERAGE_SIZE_DISTRIBUTION.png', bbox_inches='tight')
        else:
            plt.show()

