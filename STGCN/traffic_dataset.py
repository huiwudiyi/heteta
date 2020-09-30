import csv
import numpy as np
import torch as t
import torch.utils.data as Dataset

# 两种方式进行处理数据
#1. 通过文件的id进行数据的妆化
# 2. 直接将nodeid输入到 graph中

def get_adjacent_matrix(distancefidle:str,num_nodes:int,id_file:str = None,graph_type = 'connect') ->np.array:
    A = np.zeros([num_nodes,num_nodes])
    if id_file:
        with open(id_file,'r') as f:
            nodedict = {int(nodeid):idx for idx,nodeid in enumerate(f.read().strip().split('\n'))}
            if distancefidle:
                with open(distancefidle,'r') as fd:
                    fd.read()
                    reader = csv.reader(fd)
                    for data in reader:
                        if len(data) != 3:
                            continue
                        i,j,value = data[0],data[1],data[2]
                        if graph_type == 'connect':
                            A[nodedict[i],nodedict[j]] = 1.
                            A[nodedict[j],nodedict[i]] = 1.
                        elif graph_type == 'distance':
                            A[nodedict[i],nodedict[j]] = 1./float(value)
                            A[nodedict[j],nodedict[i]] = 1./float(value)
                        else:
                            raise ValueError('Graph type is connect or distance')


        return A

    with open(distancefidle,'r') as fd:
        fd.read()
        reader = csv.reader(fd)
        for data in reader:
            if len(data) != 3:
                continue
            i, j, value = data[0], data[1], data[2]
            if graph_type == 'connect':
                A[i, j] = 1.
                A[j, i] = 1.
            elif graph_type == 'distance':
                A[i, j] = 1. / float(value)
                A[j, i] = 1. / float(value)
            else:
                raise ValueError('Graph type is connect or distance')
    return A

# 获取flow_data
def get_flow_data(file,str)->np.array:
    data = np.load(file)
    # 增加维度
    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]
    return flow_data


class LoadData(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
        """
        :param data_path: list, ["graph file name" , "flow data file name"], path to save the data file names.
        :param num_nodes: int, number of nodes.
        :param divide_days: list, [ days of train data, days of test data], list to divide the original data.
        :param time_interval: int, time interval between two traffic data records (mins).
        :param history_length: int, length of history data to be used.
        :param train_mode: list, ["train", "test"].
        """

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]  # 45
        self.test_days = divide_days[1]  # 14
        self.history_length = history_length  # 6
        self.time_interval = time_interval  # 5 min

        self.one_day_length = int(24 * 60 / self.time_interval) # 计算每天时间的长度
        # 获取 矩阵
        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes)
        #  进行归一化
        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1)

    def __len__(self):
        """
        :return: length of dataset (number of samples).
        """
        #
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length # 划分的总长度
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)

        data_x = LoadData.to_tensor(data_x)  # [N, H, D]
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)  # [N, 1, D]

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data, history_length, index, train_mode):
        """
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]
        data_y = data[:, end_index]

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):#
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        # 数据进行归一化
        norm_base = LoadData.normalize_base(data, norm_dim)  # find the normalize base
        # 数据进行normalize
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            max_data: np.array
            min_data: np.array
        """
        #norm_dim  = 1
        # 时间轴上进行归一化
        #
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
        min_data = np.min(data, norm_dim, keepdims=True)
        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        #
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        #
        mid = min_data
        base = max_data - min_data
        recovered_data = data * base + mid

        return recovered_data

    @staticmethod
    def to_tensor(data):
        return t.tensor(data, dtype=t.float)


if __name__ == '__main__':
    train_data = LoadData(data_path=["data/PeMS_04/PeMS04.csv", "data/PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    print(len(train_data))
    print(train_data[0]["flow_x"].size())
    print(train_data[0]["flow_y"].size())

