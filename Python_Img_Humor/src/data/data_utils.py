from torch.utils.data import Dataset


class DataIter(Dataset):
    """ Create data iteration batching objects for
    provided matrices/vectors, for use with NN

     Input:
     - data (data to iter over)
     - target (target of prediction to iter over) """

    def __init__(self, data, target):
        self.data = data
        self.target = target

        print("self.data from DataIter shape: ", self.data.shape)
        print("self.target from DataIter shape: ", self.target.shape)

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = index.cpu().numpy()
        data = self.data[index]
        target = self.target[index]

        return data, target, index

    def __setitem__(self, index, data, target):
        self.data[index, :] = data
        self.target[index] = target

    def __len__(self):
        return len(self.data)
