from PyQt5 import QtWidgets
import sys

from gui.Ui import Ui


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = Ui()
    app.exec_()


if __name__ == '__main__':
    main()
