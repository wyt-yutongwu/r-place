import numpy as np

# disjoint-set forests using union-by-rank and path compression (sort of).
class disjoint_set:
    def __init__(self, n_elements, users=None, userembs=None):
        self.num = n_elements
        self.elts = np.empty(shape=(n_elements, 3), dtype=int)
        self.users = []
        self.userembs = []
        for i in range(n_elements):
            self.elts[i, 0] = 0  # rank
            self.elts[i, 1] = 1  # size
            self.elts[i, 2] = i  # p
            user_set = set()
            if(users is not None):
                user_set.add(users[i])
                useremb = userembs[i,:]
            self.users.append(user_set)  # size
            self.userembs.append(useremb)   # average useremb. 

    def size(self, x):
        return self.elts[x, 1]

    def get_userset(self, x):
        return self.users[x]

    def get_useremb(self, x):
        return self.userembs[x]

    def num_sets(self):
        return self.num

    def find(self, x):
        y = int(x)
        while y != self.elts[y, 2]:
            y = self.elts[y, 2]
        self.elts[x, 2] = y
        return y

    def get_comp_user(self):
        count = dict()
        emb = dict()
        label = self.elts[:,2]
        result = np.zeros((len(label), len(self.userembs[0])))
        user_embs = self.userembs
        for i in range(0, len(label)):
            lab = label[i]
            user_e = user_embs[i]
            if lab not in count:
                count[lab] = 1
                emb[lab] = user_e
            else:
                count[lab] += 1
                emb[lab] += user_e            
        for j in range(0, len(label)):
            lab = label[j]
            result[j] = emb[lab] / count[lab]
        return result

    def join(self, x, y):
        if self.elts[x, 0] > self.elts[y, 0]:
            self.elts[y, 2] = x
            self.userembs[x] = (self.elts[x,1]*self.userembs[x] + self.elts[y,1]*self.userembs[y]) / (self.elts[x, 1] + self.elts[y,1])
            self.elts[x, 1] += self.elts[y, 1]
            self.users[x] = set.union(self.users[x], self.users[y])
        else:
            self.elts[x, 2] = y
            self.userembs[y] = (self.elts[x,1]*self.userembs[x] + self.elts[y,1]*self.userembs[y]) / (self.elts[x, 1] + self.elts[y,1])
            self.elts[y, 1] += self.elts[x, 1]
            self.users[y] = set.union(self.users[x], self.users[y])
            if self.elts[x, 0] == self.elts[y, 0]:
                self.elts[y, 0] += 1
        self.num -= 1