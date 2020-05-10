from PyQt5 import QtWidgets, uic


class DialogConsole(QtWidgets.QDialog):
    def __init__(self):
        super(DialogConsole, self).__init__()
        uic.loadUi('gui/DialogConsole.ui', self)

        self.init_gui()

        self.textBrowser_console = self.findChild(QtWidgets.QTextBrowser, 'textBrowser_console')

    def init_gui(self):
        self.set_textBrowser_console()

    ###
    # textBrowser
    ###
    def set_textBrowser_console(self):
        self.textBrowser_training.setStyleSheet("background-color: black; color: white")