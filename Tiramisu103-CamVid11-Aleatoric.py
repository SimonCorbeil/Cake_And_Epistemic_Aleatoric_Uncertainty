import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from tabulate import tabulate
from src.datasets import camvid
from src import evaluate
from src import plot
from src import predict
from src import predict_video
from src.tiramisu import aleatoric_tiramisu
from src.tiramisu import tiramisu
from src.callbacks import PlotMetrics
from src.utils import history_to_results
#%matplotlib inline
# the location to save coarse training weights to
pretrain_weights = 'models/Tiramisu103-CamVid11-fine.h5'
# the location to save fine training weights to
weights_file = 'models/Tiramisu103-CamVid11-Aleatoric.h5'
# the size to crop images to for fine tune training
crop_size = (352, 480)
# the size of batches to use for training
batch_size = 1
camvid11 = camvid.CamVid(
    mapping=camvid.CamVid.load_mapping(),
    target_size=(360, 480),
    crop_size=crop_size,
    batch_size=batch_size,
    horizontal_flip=True,
    ignored_labels=['Void'],
    y_repeats=1,
)
generators = camvid11.generators()

# get the next X, y training tuple
X, y = next(generators['train'])
# transform the onehot vector to an image
y = camvid11.unmap(y[0])
# plot the images
_ = plot.plot(X=X[0], y=y[0], order=['X', 'y'])

# build the model for the image shape and number of labels
model = aleatoric_tiramisu.aleatoric_tiramisu((*crop_size, 3), camvid11.n,
    class_weights=camvid11.class_mask,
    learning_rate=1e-4,
    weights_file=pretrain_weights,
)
model.summary()
callbacks = [
    EarlyStopping(monitor='val_aleatoric_loss', patience=10),
    ModelCheckpoint(weights_file,
        monitor='val_aleatoric_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    ),
    PlotMetrics(),
]
# fit the model with the data.
history = model.fit_generator(generators['train'],
    epochs=10,
    steps_per_epoch=int(367 / batch_size),
    validation_data=generators['val'],
    validation_steps=101,
    callbacks=callbacks,
    verbose=0,
)

#history_to_results(history)

# model.load_weights(weights_file)

# metrics = evaluate.evaluate(model, generators['test'], 233,
#     mask=camvid11.class_mask, 
#     code_map=camvid11.discrete_to_label_map,
# )
# metrics.to_csv(weights_file + '.csv')
# metrics

# print(tabulate(metrics, tablefmt='pipe', headers=('Metric', 'Value')))

# X, y, p, s = predict.predict_aleatoric(model, generators['train'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['train'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['train'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['train'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['val'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['val'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['val'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['val'], camvid11)
# _ = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])

# X, y, p, s = predict.predict_aleatoric(model, generators['test'], camvid11)
# fig = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])
# fig.savefig('img/tiramisu-bayesian/aleatoric/0.png', transparent=True, bbox_inches='tight')

# X, y, p, s = predict.predict_aleatoric(model, generators['test'], camvid11)
# fig = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])
# fig.savefig('img/tiramisu-bayesian/aleatoric/1.png', transparent=True, bbox_inches='tight')

# X, y, p, s = predict.predict_aleatoric(model, generators['test'], camvid11)
# fig = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])
# fig.savefig('img/tiramisu-bayesian/aleatoric/2.png', transparent=True, bbox_inches='tight')

# X, y, p, s = predict.predict_aleatoric(model, generators['test'], camvid11)
# fig = plot.plot(X=X[0], y=y[0], y_pred=p[0], aleatoric=s[0], order=['X', 'y', 'y_pred', 'aleatoric'])
# fig.savefig('img/tiramisu-bayesian/aleatoric/3.png', transparent=True, bbox_inches='tight')

# video_file = '0005VD.mp4'
# video_path = camvid.videos.abs_path(video_file)
# out_path = 'img/tiramisu-bayesian/aleatoric/{}'.format(video_file)

# predict_video.predict_video(video_path, out_path, camvid11, model, predict.predict_aleatoric)