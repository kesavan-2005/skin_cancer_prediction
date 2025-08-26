# ============================
# Multimodal Skin Lesion Model
# - No EarlyStopping
# - Grad-CAM added
# ============================

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# ------------------ Paths ------------------
DATA_CSV = r'D:\skin_cancer\data\processed\data_with_paths.csv'
MODEL_PATH = r'D:\skin_cancer\fmodel\trained_multimodal_model.h5'
LABEL_ENCODER_PATH = r'D:\skin_cancer\fmodel\label_encoder.pkl'
SCALER_PATH = r'D:\skin_cancer\fmodel\age_scaler.pkl'
DX_TYPE_ENCODER_PATH = r'D:\skin_cancer\fmodel\dx_type_encoder.pkl'
SEX_ENCODER_PATH = r'D:\skin_cancer\fmodel\sex_encoder.pkl'
LOC_ENCODER_PATH = r'D:\skin_cancer\fmodel\localization_encoder.pkl'

MODEL_DIR = os.path.dirname(MODEL_PATH)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ Load Data ------------------
df = pd.read_csv(DATA_CSV)

# Sanity: ensure required columns exist
required_cols = {'image_path', 'diagnosis', 'dx_type', 'sex', 'localization', 'age'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in DATA_CSV: {missing}")

# Encode target labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['diagnosis'])
with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(le, f)

# Encode categorical features (save encoders for inference)
dx_type_le = LabelEncoder()
sex_le = LabelEncoder()
loc_le = LabelEncoder()

df['dx_type_enc'] = dx_type_le.fit_transform(df['dx_type'].astype(str))
df['sex_enc'] = sex_le.fit_transform(df['sex'].astype(str))
df['localization_enc'] = loc_le.fit_transform(df['localization'].astype(str))

with open(DX_TYPE_ENCODER_PATH, 'wb') as f:
    pickle.dump(dx_type_le, f)
with open(SEX_ENCODER_PATH, 'wb') as f:
    pickle.dump(sex_le, f)
with open(LOC_ENCODER_PATH, 'wb') as f:
    pickle.dump(loc_le, f)

# Scale numerical feature (save scaler)
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']].fillna(df['age'].median()))
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# ------------------ Split ------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

num_classes = df['label'].nunique()
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # adjust if you hit OOM

# ------------------ Data Augmentation for Training ------------------
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Optional: uncomment for a bit more variety
    # image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    return image

# ------------------ TF Dataset Functions ------------------
def load_image_and_features(row, augment=False):
    image = tf.io.read_file(row['image_path'])
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)

    if augment:
        image = augment_image(image)

    image = preprocess_input(image)  # MobileNetV2 expects [-1, 1]

    # Build tabular feature tensor (float32)
    features = tf.stack([
        tf.cast(row['dx_type_enc'], tf.float32),
        tf.cast(row['age_scaled'], tf.float32),
        tf.cast(row['sex_enc'], tf.float32),
        tf.cast(row['localization_enc'], tf.float32)
    ])

    label = tf.one_hot(tf.cast(row['label'], tf.int32), depth=num_classes)
    return (image, features), label

def df_to_dataset(dataframe, shuffle=True, batch_size=32, augment=False):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    ds = ds.map(lambda x: load_image_and_features(x, augment=augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ------------------ Prepare Datasets ------------------
train_ds = df_to_dataset(train_df, shuffle=True, batch_size=BATCH_SIZE, augment=True)
val_ds = df_to_dataset(val_df, shuffle=False, batch_size=BATCH_SIZE, augment=False)

# ------------------ Build Model ------------------
image_input = Input(shape=(224, 224, 3), name="image_input")
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=image_input)
x = GlobalAveragePooling2D(name="gap")(base_model.output)

tab_input = Input(shape=(4,), name="tabular_input")
t = Dense(64, activation='relu')(tab_input)
t = BatchNormalization()(t)
t = Dropout(0.4)(t)
t = Dense(32, activation='relu')(t)
t = Dropout(0.3)(t)

combined = Concatenate(name="fusion_concat")([x, t])
combined = Dense(128, activation='relu')(combined)
output = Dense(num_classes, activation='softmax', name='predictions')(combined)

model = Model(inputs=[image_input, tab_input], outputs=output)

# Freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

# ------------------ Focal Loss Definition ------------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        # Return mean loss per example
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed

# Compile model with focal loss
model.compile(optimizer=Adam(1e-4), loss=focal_loss(gamma=2., alpha=0.25), metrics=["accuracy"])

# ------------------ Class Weights ------------------
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weights = dict(enumerate(class_weights_array))
print("📊 Class Weights:", class_weights)

# ------------------ Callbacks (No EarlyStopping) ------------------
checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-7, verbose=1)

# ------------------ Train Head (No EarlyStopping) ------------------
EPOCHS_HEAD = 35
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

# ------------------ Fine-tune ------------------
# Unfreeze more than last 10 if you wish; here: last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss=focal_loss(gamma=2., alpha=0.25), metrics=["accuracy"])

EPOCHS_FT = 30
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr],
    verbose=1
)

# ------------------ Plot History ------------------
def plot_history(histories):
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    for h in histories:
        acc += h.history.get('accuracy', [])
        val_acc += h.history.get('val_accuracy', [])
        loss += h.history.get('loss', [])
        val_loss += h.history.get('val_loss', [])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history([history1, history2])

# ------------------ Load Best Weights (from checkpoint) ------------------
best_model = load_model(
    MODEL_PATH,
    custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)}
)

# ------------------ Final Evaluation ------------------
val_preds = best_model.predict(val_ds, verbose=1)
y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
y_pred = np.argmax(val_preds, axis=1)
y_true = np.argmax(y_true, axis=1)

from sklearn.metrics import classification_report
print("\n📊 Classification Report (Best Checkpoint):\n")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ==============================================================
#                         Grad-CAM
# ==============================================================

# Grad-CAM for a multi-input model. We target the last conv layer of MobileNetV2: "Conv_1"
LAST_CONV_LAYER_NAME = "Conv_1"  # in MobileNetV2

def make_gradcam_heatmap(img_array, tab_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME, class_index=None):
    """
    img_array: preprocessed image tensor of shape (1, 224, 224, 3)
    tab_array: tabular tensor of shape (1, 4)
    class_index: optional target class int; if None, uses predicted class
    returns: heatmap (H, W) in range [0, 1]
    """
    # Build a model that maps inputs -> (last conv outputs, predictions)
    image_input_tensor = model.get_layer('image_input').input
    tab_input_tensor = model.get_layer('tabular_input').input

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(
        inputs=[image_input_tensor, tab_input_tensor],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, tab_array], training=False)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    # Compute gradients of the top predicted class wrt the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    # Global average pooling across width and height to get importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # ReLU
    heatmap = tf.nn.relu(heatmap)
    # Normalize to [0, 1]
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def load_and_preprocess_image(img_path):
    raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img

def get_tab_tensor(row):
    feats = np.array([
        float(row['dx_type_enc']),
        float(row['age_scaled']),
        float(row['sex_enc']),
        float(row['localization_enc'])
    ], dtype=np.float32)
    return feats

def show_gradcam(img_path, heatmap, alpha=0.35, cmap='jet', save_path=None):
    """
    Overlay heatmap on original image (non-preprocessed).
    """
    # Load original image for display
    raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, IMG_SIZE).numpy().astype('uint8')

    # Rescale heatmap to 0-255
    heatmap_rescaled = np.uint8(255 * heatmap)

    # Colorize heatmap using matplotlib
    import matplotlib.cm as cm
    colormap = cm.get_cmap(cmap)
    colored_hm = colormap(heatmap_rescaled)  # (H, W, 4) RGBA
    colored_hm = np.uint8(colored_hm[:, :, :3] * 255)

    # Overlay
    overlay = np.uint8(alpha * colored_hm + (1 - alpha) * img)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    plt.imshow(heatmap, cmap=cmap)

    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.axis('off')
    plt.imshow(overlay)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

# -------- Example: Generate Grad-CAM for a few validation samples --------
def gradcam_example(best_model, df_subset, indices=None, max_examples=3, target_class=None):
    """
    df_subset: a DataFrame with validation rows
    indices: list of row indices (relative to df_subset) to visualize; if None, use first N
    target_class: optional int for class index; if None uses predicted class
    """
    if indices is None:
        indices = list(range(min(max_examples, len(df_subset))))

    for idx in indices:
        row = df_subset.iloc[idx]
        img_path = row['image_path']
        # Build input tensors
        img = load_and_preprocess_image(img_path).numpy()
        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
        tab = get_tab_tensor(row)
        tab_batch = np.expand_dims(tab, axis=0).astype(np.float32)

        # Predict & Grad-CAM
        preds = best_model.predict([img_batch, tab_batch], verbose=0)
        pred_class = int(np.argmax(preds[0]))
        class_idx = target_class if target_class is not None else pred_class

        heatmap = make_gradcam_heatmap(img_batch, tab_batch, best_model,
                                       last_conv_layer_name=LAST_CONV_LAYER_NAME,
                                       class_index=class_idx)
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"Predicted: {le.classes_[pred_class]} (prob={preds[0][pred_class]:.3f})")
        if target_class is not None:
            print(f"Grad-CAM target class: {le.classes_[target_class]} ({target_class})")
        show_gradcam(img_path, heatmap, alpha=0.35, cmap='jet')

# Run a small Grad-CAM demo for the first 3 validation images:
gradcam_example(best_model, val_df.reset_index(drop=True), max_examples=3)
