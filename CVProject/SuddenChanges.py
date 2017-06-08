import numpy as np
from collections import deque

class SuddenChanges:

    def __init__(self, maxlen, N_vars, var_type, threshold=25):
        self.var_type = var_type
        self.threshold = threshold
        self.que = [None] * N_vars
        for n in range(N_vars):
            self.que[n] = deque(maxlen=maxlen)

    def CheckChange(self, *var):
        for index, v in enumerate(var):
            self.que[index].append(v)
            if len(self.que[index]) == self.que[index].maxlen:
                std = np.std(self.que[index])
                if std > self.threshold:
                    print("[FAIL] ({}) index [{}] and std {} > threshold {}".format(self.var_type, index, std, self.threshold))
                    return True
        return False

    def ClearQue(self):
        for index in range(len(self.que)):
            self.que[index].clear()