import matplotlib.pyplot as plt


class Graphs:
    @staticmethod
    def line_graph():
        plt.plot([1, 2, 3, 4])
        plt.ylabel('some numbers')
        plt.show()