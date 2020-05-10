import matplotlib.pyplot as plt


class Utils:
    @staticmethod
    def display_image(img_path):
        plt.figure(figsize=(6, 6))
        plt.imshow(img_path)
        plt.axis('off')
        plt.show()
