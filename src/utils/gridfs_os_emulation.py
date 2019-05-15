import os
import glob


class OsFs():
    """
    Emulate filesystem API of pymongo (gridfs) on with standard os module
    (since this project used gridfs during development).
    Note: Only emulates the commands required in this project
    """

    def __init__(self, base_path):
        """

        :param base_path: all the data will be saved to base_path directory
        """

        os.makedirs(base_path, exist_ok=True)
        self.base_path = base_path

    def new_file(self, filename):
        full_filename = os.path.join(self.base_path, filename)
        dirname = os.path.dirname(full_filename)
        os.makedirs(dirname, exist_ok=True)
        return open(full_filename, 'wb')

    def list(self):
        print (self.base_path)
        return glob.glob(self.base_path + '/**/*', recursive=True)

    def get_last_version(self, filename):
        """ return an opened file for reading. Note: # os doesn't support versioning """

        full_filename = os.path.join(self.base_path, filename)
        fp = open(full_filename, 'rb')
        vars(fp)['_file'] = {}
        vars(fp)['_file']['_id'] = -1 # set to -1 since os doesn't support versioning

        return fp
