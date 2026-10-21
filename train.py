"""
Train the MobileNetV2 pneumonia classifier and export the files app.py needs:
    models/pneumonia_classifier.h5
    models/labels.txt

Dataset layout (Kaggle "Chest X-Ray Images (Pneumonia)"):
    chest_xray/train/{NORMAL,PNEUMONIA}
    chest_xray/test/{NORMAL,PNEUMONIA}

Usage:
    python train.py --data_dir chest_xray
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

IMG = (224, 224)


def load_ds(path, batch, subset=None, shuffle=True):
    kw = dict(image_size=IMG, batch_size=batch, label_mode='categorical',
              color_mode='rgb', shuffle=shuffle, seed=42)
    if subset:
        kw.update(validation_split=0.15, subset=subset)
    return tf.keras.utils.image_dataset_from_directory(path, **kw)


def main(a):
    train_ds = load_ds(os.path.join(a.data_dir, 'train'), a.batch_size, 'training')
    val_ds = load_ds(os.path.join(a.data_dir, 'train'), a.batch_size, 'validation')
    test_ds = load_ds(os.path.join(a.data_dir, 'test'), a.batch_size, shuffle=False)
    class_names = train_ds.class_names  # alphabetical: NORMAL, PNEUMONIA
    print('Classes:', class_names)

    # Class weights (pneumonia images outnumber normal ~3:1)
    counts = np.zeros(len(class_names))
    for _, y in train_ds.unbatch():
        counts[int(np.argmax(y.numpy()))] += 1
    class_w = {i: counts.sum() / (len(counts) * c) for i, c in enumerate(counts)}
    print('Class weights:', class_w)

    augment = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.15),
    ])
    scale = lambda x, y: (x / 127.5 - 1.0, y)  # same as util.classify
    auto = tf.data.AUTOTUNE
    train_p = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=auto) \
                      .map(scale, num_parallel_calls=auto).prefetch(auto)
    val_p = val_ds.map(scale, num_parallel_calls=auto).prefetch(auto)
    test_p = test_ds.map(scale, num_parallel_calls=auto).prefetch(auto)

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax'),
    ])

    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    ckpt = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=3, restore_best_weights=True)

    # Stage 1: train the new head
    model.compile(tf.keras.optimizers.Adam(1e-3), 'categorical_crossentropy', metrics=metrics)
    model.fit(train_p, validation_data=val_p, epochs=a.epochs,
              class_weight=class_w, callbacks=[ckpt])

    # Stage 2: fine-tune the top of the backbone
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-5), 'categorical_crossentropy', metrics=metrics)
    model.fit(train_p, validation_data=val_p, epochs=a.fine_tune_epochs,
              class_weight=class_w, callbacks=[ckpt])

    # Evaluate
    y_true = np.concatenate([np.argmax(y.numpy(), 1) for _, y in test_p])
    y_pred = np.argmax(model.predict(test_p, verbose=0), 1)
    print('\n=== Test set ===')
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(confusion_matrix(y_true, y_pred))

    # Export for app.py
    os.makedirs('models', exist_ok=True)
    model.save('models/pneumonia_classifier.h5')
    with open('models/labels.txt', 'w') as f:
        for i, n in enumerate(class_names):
            f.write(f'{i} {n}\n')
    print('Saved models/pneumonia_classifier.h5 and models/labels.txt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--fine_tune_epochs', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=32)
    main(p.parse_args())