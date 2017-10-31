import numpy as np
class item():
    def __init__(self, row_length, row1, row2, result):
        self.row1 = row1
        self.row2 = row2
        self.id = row_length*row1 + row2
        self.result = np.array(result)
