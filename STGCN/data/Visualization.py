import numpy as np
import matplotlib.pyplot as plt

def get_flowData(file):
    flow_data = np.load(file)
    print([key for key in flow_data.keys()])
    flow_data = flow_data['data']
    return flow_data

if __name__ == '__main__':
    flow_data = get_flowData('PeMS_04/PeMS04.npz')
    print(flow_data.shape)