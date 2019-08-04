from dataset import Dataset
import numpy as np
arr=np.random.rand(3,4);
label=np.ones([4,1])
initialize_iteration=1000
batch_size=2
embedding=Dataset(train_x=arr,train_y=label)
next_ = embedding.gen_next_batch(batch_size=batch_size, is_train_set=True, iteration=initialize_iteration)
for iter_, (batch_x, _, _) in enumerate(next_):
    print(iter_)
