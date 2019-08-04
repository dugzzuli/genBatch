import numpy as np

def batch_generator(all_data , batch_size, shuffle=True):
    """
    :param all_data : all_data整个数据集
    :param batch_size: batch_size表示每个batch的大小
    :param shuffle: 每次是否打乱顺序
    :return:
    """
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    print("data_size: ", data_size)
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]


# 输入x表示有23个样本，每个样本有两个特征
# 输出y表示有23个标签，每个标签取值为0或1
x = np.random.random(size=[23, 2])
y = np.random.randint(2, size=[23,1])

batch_size = 5
batch_gen = batch_generator([x, y],  batch_size)
for i in range(20):
    batch_x, batch_y = next(batch_gen)
    print(batch_x, batch_y)
