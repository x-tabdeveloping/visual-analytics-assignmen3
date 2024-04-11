from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from keras.applications import VGG16
from keras.layers import BatchNormalization, Dense, Dropout, Flatten
from keras.metrics import CategoricalAccuracy
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm


def three_way_split(
    x,
    y,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratified: bool = True,
) -> tuple:
    """Splits dataset to train, val and test sets."""
    x, x_test, y, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def pad_to_square(im: Image, size: int = 224) -> Image:
    """Pads an image to a given sized square."""
    thumbnail = im.copy()
    thumbnail.thumbnail((size, size))
    width, height = thumbnail.size
    if width == height:
        return thumbnail
    elif width > height:
        result = Image.new(thumbnail.mode, (width, width))
        result.paste(thumbnail, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(thumbnail.mode, (height, height))
        result.paste(thumbnail, ((height - width) // 2, 0))
        return result


def freeze(model):
    """Freezes all layers in a model (makes them non-trainable)"""
    for layer in model.layers:
        layer.trainable = False


out_path = Path("out")
out_path.mkdir(exist_ok=True)

print("Loading and preprocessing images.")
input_files = list(Path("dat/Tobacco3482-jpg").glob("**/*.jpg"))
images = []
labels = []
for file in tqdm(input_files):
    with Image.open(file) as im:
        im = pad_to_square(im, 224).convert("RGB")
        im_arr = np.array(im)
        im_arr = (im_arr / 255).astype(np.float32)
        images.append(im_arr)
        labels.append(Path(file).parent.stem)
images = np.stack(images)
binarizer = LabelBinarizer()
y = binarizer.fit_transform(labels).astype(np.float32)
classes = binarizer.classes_

print("Splitting dataset.")
x_train, x_val, x_test, y_train, y_val, y_test = three_way_split(images, y)

print("Building Model.")
pretrained = VGG16(input_shape=(images.shape[1:]))
freeze(pretrained)
model = Sequential(
    [
        pretrained,
        Flatten(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(len(classes), activation="softmax"),
    ]
)
model.compile(
    optimizer=Adam(learning_rate=1e-03),
    loss="categorical_crossentropy",
    metrics=[CategoricalAccuracy()],
)


N_EPOCHS = 10
print("Model training.")
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=N_EPOCHS,
    validation_data=(x_val, y_val),
)

print("Making figures.")
loss_curve = go.Figure()
loss_curve.add_trace(
    go.Scatter(name="Training Loss", y=history.history["loss"], x=np.arange(N_EPOCHS))
)
loss_curve.add_trace(
    go.Scatter(
        name="Validation Loss", y=history.history["val_loss"], x=np.arange(N_EPOCHS)
    )
)
loss_curve.update_layout(width=1024, height=1024)
loss_curve.update_layout(
    xaxis_title="Epoch", yaxis_title="Loss", title="Loss of the Model"
)
loss_curve.write_image(out_path.joinpath("loss_curve.png"))

print("Making classification report")
y_pred = model.predict(x_test)
y_pred = binarizer.inverse_transform(y_pred)
y_true = binarizer.inverse_transform(y_test)
report = classification_report(y_true, y_pred)
with out_path.joinpath("classification_report.txt").open("w") as out_file:
    out_file.write(report)

print("DONE.")
