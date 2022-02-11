import os


class csv_writer:

    def __init__(self, file_name, col_names):
        self.file_string = ""
        self.file_name = file_name

        for i in range(len(col_names) - 1):
            self.file_string += str(col_names[i]) + ","

        self.file_string += str(col_names[len(col_names) - 1])
        self.file_string += "\n"




    def add_row(self, data_list):

        for i in range(len(data_list) - 1):
            self.file_string += str(data_list[i]) + ","

        self.file_string += str(data_list[len(data_list) - 1])
        self.file_string += "\n"



    def write_to_file(self):
        f = open(self.file_name, "w")

        f.write(self.file_string)

        f.close()
