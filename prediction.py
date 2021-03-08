# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
# %%
data = pd.read_csv('./cleanData/all.csv',index_col='time',parse_dates=['time'],infer_datetime_format=True)
delivery_note = pd.read_csv('./rawData/delivery_note.csv',sep='\t')
delivery_note = delivery_note.set_index(['id'])
delivery_add = pd.DataFrame([['bre0','lightning count','No'],['onoff','lightning onoff','No']],columns=['id','name','unit'])
delivery_add = delivery_add.set_index(['id'])
delivery_note = delivery_note.append(delivery_add)

#data = data.reindex(columns=sorted(data.columns))
data = data.reindex(columns=(list([a for a in data.columns if a not in ['brecloz0','brefarz0','bre0','onoff'] ]) + ['brecloz0','brefarz0','bre0','onoff'] ))
data



# %%
colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
def show_raw_visualization(data):
    time_data = data.index
    fig, axes = plt.subplots(
        nrows=8, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(data.columns)):
        key = data.columns[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{}({}) - {}".format(delivery_note.loc[key]['name'],delivery_note.loc[key]['unit'], key),
            rot=25,
        )
        ax.legend(key)
    plt.tight_layout()


show_raw_visualization(data)


# %%

def show_heatmap(data,method='pearson'):
    plt.matshow(data.corr(method=method))
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


show_heatmap(data,'pearson')

show_heatmap(data,'kendall')

show_heatmap(data,'spearman')

# %%

#train,test = train_test_split(data)
#mean = train.mean(axis=0)
#std = train.std(axis=0)
#train_ = (train-mean)/std
#test_ = (test-mean)/std

# %%

split_fraction = 0.715
train_split = int(split_fraction * int(data.shape[0]))
step = 6

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 10


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std
data
#data_ = pd.DataFrame(normalize(data.values,train_split),columns=data.columns)
data_ = pd.DataFrame(normalize(data.values,train_split))
data_
#train_split = 10
train_data = data_.loc[0:train_split-1]
val_data = data_.loc[train_split:]

start = past + future
end = start + train_split

X_train = train_data[[i for i in range(12)]].values
y_train = train_data[[14]]
X_train.shape
y_train.shape
sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(X_train,y_train,sequence_length=sequence_length,batch_size=batch_size)
# %%

X_val = val_data[[i for i in range(12)]].values
y_val = val_data[[14]]
dataset_val = keras.preprocessing.timeseries_dataset_from_array(X_val,y_val,sequence_length=sequence_length,batch_size=batch_size)
# %%

for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

# %%

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

# %%

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

# %%

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")
