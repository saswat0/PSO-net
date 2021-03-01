import h5py

def data_gen(data_store_data, data_store_label, batch_size):
    while True:
        for i in range(0, data_store_data.shape[0], batch_size):
            data_batch = data_store_data[i:i+batch_size,:]
            label_batch = data_store_label[i:i+batch_size,:]
            yield data_batch, label_batch
