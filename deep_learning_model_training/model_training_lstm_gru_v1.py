# Updated script using Stratified K-Fold Cross-Validation (5 folds) with full PNG exports and feature importance
import os
import json
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import datetime
from scipy.stats import skew # type: ignore

from sklearn.model_selection import StratifiedKFold # type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, f1_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

import tensorflow as tf # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Attention, Concatenate, Permute, Multiply, Dropout, LayerNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from tensorflow.keras import regularizers # type: ignore

def load_json_data(base_path):
    X, y = [], []
    for label in ['truthful', 'deceitful']:
        folder = os.path.join(base_path, label)
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), 'r') as f:
                data = json.load(f)
                trajectory = np.array(data['mouseMovements'], dtype=np.float32)
                acc = np.array(data['accelerations'], dtype=np.float32)
                jerk = np.array(data['jerks'], dtype=np.float32)
                curvature = np.array(data['curvatures'], dtype=np.float32)
                timestamps = np.array(data['timestamps'], dtype=np.float32)

                velocity = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)) / np.diff(timestamps)
                velocity = np.insert(velocity, 0, 0.0)

                features = np.stack([trajectory[:, 0], trajectory[:, 1], velocity, acc, curvature, jerk], axis=1)

                def extract_stats(array):
                    return [np.mean(array), np.std(array), np.min(array), np.max(array), skew(array) if len(array) > 2 else 0.0]

                seq_stats = []
                for dim in [velocity, acc, curvature, jerk]:
                    seq_stats.extend(extract_stats(dim))

                total_time = data.get("totalTime", 0)
                pause_count = len(data.get("pausePoints", []))
                pause_time = total_time * 0.1 * pause_count
                time_to_first = timestamps[0] if len(timestamps) > 0 else 0

                meta = [total_time, time_to_first, pause_time, pause_count] + seq_stats

                X.append((features, meta))
                y.append(0 if label == 'truthful' else 1)
    return X, y

def preprocess_data(X_raw):
    X_seq = tf.keras.preprocessing.sequence.pad_sequences(
        [x[0] for x in X_raw], padding='post', dtype='float32')
    X_meta = np.array([x[1] for x in X_raw], dtype='float32')
    return X_seq, X_meta

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return alpha * tf.keras.backend.pow(1 - p_t, gamma) * bce
    return loss

def build_model(input_seq_shape, input_meta_shape):
    seq_input = Input(shape=input_seq_shape)
    x = LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(seq_input)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = GRU(16, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    attention = Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(16)(attention)
    attention = Permute([2, 1])(attention)

    x = Multiply()([x, attention])
    x = tf.keras.layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)

    meta_input = Input(shape=(input_meta_shape,))
    concat = Concatenate()([x, meta_input])
    dense = Dense(64, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[seq_input, meta_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=focal_loss(), metrics=['accuracy'])
    return model

def save_confusion_matrix(y_true, y_pred, fold):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['truthful', 'deceitful']
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Fold {fold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"confusion_matrix_fold_{fold}.png")
    plt.close()
    return cm

def save_roc_curve(y_true, y_prob, fold):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='orange')
    plt.title(f"ROC Curve (Fold {fold})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"roc_curve_fold_{fold}.png")
    plt.close()

def plot_training_history(history, fold):
    df = pd.DataFrame(history.history)
    df.to_csv(f"training_log_fold_{fold}.csv", index=False)
    plt.plot(df['loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Val Loss')
    plt.title(f'Training/Validation Loss (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"training_validation_loss_fold_{fold}.png")
    plt.close()

def save_feature_correlations(X_meta, feature_names):
    df = pd.DataFrame(X_meta, columns=feature_names)
    corr = df.corr().abs()
    mean_corr = corr.mean().sort_values(ascending=False)
    top_features = mean_corr.head(10).index.tolist()
    corr_subset = df[top_features].corr()
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_subset, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', center=0)
    plt.title("Top 10 Feature Correlations")
    plt.savefig("feature_correlation_matrix.png")
    plt.close()

def save_feature_importance(model, X_seq_val, X_meta_val, y_val, feature_names, fold):
    baseline_acc = accuracy_score(y_val, (model.predict([X_seq_val, X_meta_val]) > 0.5).astype(int))
    importances = []
    for i in range(X_meta_val.shape[1]):
        X_meta_permuted = X_meta_val.copy()
        np.random.shuffle(X_meta_permuted[:, i])
        acc = accuracy_score(y_val, (model.predict([X_seq_val, X_meta_permuted]) > 0.5).astype(int))
        importances.append(baseline_acc - acc)
    sorted_idx = np.argsort(importances)
    importances_array = np.array(importances)[sorted_idx]
    features_sorted = np.array(feature_names)[sorted_idx]
    important_mask = importances_array > 0.005
    plt.figure(figsize=(10, 6))
    plt.barh(np.arange(sum(important_mask)), importances_array[important_mask], color='steelblue')
    plt.yticks(np.arange(sum(important_mask)), features_sorted[important_mask])
    plt.title(f"Top Feature Importances (Fold {fold})")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"feature_importance_fold_{fold}.png")
    plt.close()

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
X_raw, y = load_json_data('data')
X_seq, X_meta = preprocess_data(X_raw)
y = np.array(y)

scaler = StandardScaler()
X_meta = scaler.fit_transform(X_meta)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

feature_names = [
    'total_time','time_to_first_movement', 'total_pause_time', 'pause_count',
    'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max', 'velocity_skew',
    'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max', 'acceleration_skew',
    'curvature_mean', 'curvature_std', 'curvature_min', 'curvature_max', 'curvature_skew',
    'jerk_mean', 'jerk_std', 'jerk_min', 'jerk_max', 'jerk_skew'
]
save_feature_correlations(X_meta, feature_names)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq, y)):
    print(f"\n--- Fold {fold + 1} ---")
    X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
    X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model(X_seq.shape[1:], X_meta.shape[1])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(f'model_fold_{fold+1}.h5', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        [X_seq_train, X_meta_train], y_train,
        validation_data=([X_seq_val, X_meta_val], y_val),
        epochs=30, batch_size=32, callbacks=callbacks, verbose=0
    )

    plot_training_history(history, fold+1)
    y_val_prob = model.predict([X_seq_val, X_meta_val])
    y_val_pred = (y_val_prob > 0.5).astype(int)

    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average='macro')
    cm = save_confusion_matrix(y_val, y_val_pred, fold+1)
    save_roc_curve(y_val, y_val_prob, fold+1)
    save_feature_importance(model, X_seq_val, X_meta_val, y_val, feature_names, fold+1)

    print(f"Fold {fold+1} Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    fold_results.append({
        'fold': fold+1,
        'accuracy': acc,
        'f1_macro': f1,
        'tp': cm[1, 1], 'fp': cm[0, 1],
        'tn': cm[0, 0], 'fn': cm[1, 0]
    })

results_df = pd.DataFrame(fold_results)
results_df.loc['avg'] = results_df.mean(numeric_only=True)
results_df.to_csv(f"cv_results_{run_id}.csv", index=False)
print("\nAverage Results:")
print(results_df.tail(1))


