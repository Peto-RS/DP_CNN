import os


class Utils:
    @staticmethod
    def create_folder_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    @staticmethod
    def list_files_with_extensions(path, extensions):
        files = []

        if os.path.exists(path):
            for file in os.listdir(path):
                for ext in extensions:
                    if file.endswith(ext):
                        files.append(file)

        return files

    @staticmethod
    def list_folders(path):
        folders = []

        if os.path.exists(path):
            for filename in os.listdir(path):
                if os.path.isdir(os.path.join(path, filename)):
                    folders.append(filename)

        return folders
