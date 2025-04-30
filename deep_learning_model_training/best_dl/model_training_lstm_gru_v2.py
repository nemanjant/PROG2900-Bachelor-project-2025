# Updated script using Stratified K-Fold Cross-Validation (5 folds) with full PNG exports
# Including: Confusion Matrix, ROC Curve, Calibration Curve, Precision-Recall Curve,
# Training Curves (Loss & Accuracy), Feature Correlations, and Feature Importances
import os
import json
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import skew  # type: ignore


from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.metrics import ( # type: ignore
    confusion_matrix, accuracy_score, roc_curve, auc,
    f1_score, precision_recall_curve, average_precision_score
)  # type: ignore
from sklearn.calibration import calibration_curve  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, LSTM, GRU, Dense, Concatenate, Permute,
    Multiply, Dropout, LayerNormalization
)  # type: ignore
from tensorflow.keras.callbacks import ( # type: ignore
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)  # type: ignore
from tensorflow.keras import regularizers  # type: ignore
import datetime

# -- Data Loading & Preprocessing ------------------------------------------------

def load_json_data(base_path):
    X, y = [], []
    for label in ['truthful', 'deceitful']:
        folder = os.path.join(base_path, label)
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), 'r') as f:
                data = json.load(f)
                traj = np.array(data['mouseMovements'], dtype=np.float32)
                acc = np.array(data['accelerations'], dtype=np.float32)
                jerk = np.array(data['jerks'], dtype=np.float32)
                curvature = np.array(data['curvatures'], dtype=np.float32)
                timestamps = np.array(data['timestamps'], dtype=np.float32)

                velocity = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1)) / np.diff(timestamps)
                velocity = np.insert(velocity, 0, 0.0)

                features = np.stack([traj[:, 0], traj[:, 1], velocity, acc, curvature, jerk], axis=1)

                def extract_stats(arr):
                    return [
                        np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
                        skew(arr) if len(arr) > 2 else 0.0
                    ]

                stats = []
                for arr in [velocity, acc, curvature, jerk]:
                    stats.extend(extract_stats(arr))

                total_time = data.get("totalTime", 0)
                pause_count = len(data.get("pausePoints", []))
                pause_time = total_time * 0.1 * pause_count
                time_to_first = timestamps[0] if len(timestamps) > 0 else 0

                meta = [total_time, time_to_first, pause_time, pause_count] + stats
                X.append((features, meta))
                y.append(0 if label == 'truthful' else 1)
    return X, y


def preprocess_data(X_raw):
    X_seq = tf.keras.preprocessing.sequence.pad_sequences(
        [x[0] for x in X_raw], padding='post', dtype='float32'
    )
    X_meta = np.array([x[1] for x in X_raw], dtype='float32')
    return X_seq, X_meta

# -- Loss & Model Definition ------------------------------------------------------

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

    # Simple attention mechanism
    att = Dense(1, activation='tanh')(x)
    att = tf.keras.layers.Flatten()(att)
    att = tf.keras.layers.Activation('softmax')(att)
    att = tf.keras.layers.RepeatVector(16)(att)
    att = Permute([2, 1])(att)
    x = Multiply()([x, att])
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    meta_input = Input(shape=(input_meta_shape,))
    concat = Concatenate()([x, meta_input])
    d = Dense(64, activation='relu')(concat)
    out = Dense(1, activation='sigmoid')(d)

    model = Model(inputs=[seq_input, meta_input], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(), metrics=['accuracy']
    )
    return model

# -- Plotting & Saving Functions --------------------------------------------------

def save_confusion_matrix(y_true, y_pred, fold):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=['truthful','deceitful'], columns=['truthful','deceitful'])
    plt.figure(figsize=(6,5))
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
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC Curve (Fold {fold})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"roc_curve_fold_{fold}.png")
    plt.close()


def save_precision_recall_curve(y_true, y_prob, fold):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.2f})")
    plt.title(f"Precision-Recall Curve (Fold {fold})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"precision_recall_curve_fold_{fold}.png")
    plt.close()


def save_calibration_curve(y_true, y_prob, fold, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure()
    plt.plot(mean_pred, frac_pos, 's-', label='Calibration curve')
    plt.plot([0,1],[0,1],'k--', label='Perfectly calibrated')
    plt.title(f"Calibration Curve (Fold {fold})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.savefig(f"calibration_curve_fold_{fold}.png")
    plt.close()


def plot_training_history(history, fold):
    df = pd.DataFrame(history.history)
    df.to_csv(f"training_log_fold_{fold}.csv", index=False)
    # Loss Curve
    plt.figure()
    plt.plot(df['loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Val Loss')
    plt.title(f'Training/Validation Loss (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"training_validation_loss_fold_{fold}.png")
    plt.close()
    # Accuracy Curve
    plt.figure()
    plt.plot(df['accuracy'], label='Train Acc')
    plt.plot(df['val_accuracy'], label='Val Acc')
    plt.title(f'Training/Validation Accuracy (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"training_validation_accuracy_fold_{fold}.png")
    plt.close()

# -- Feature Correlation & Importance ---------------------------------------------

def save_feature_correlations(X_meta, feature_names):
    df = pd.DataFrame(X_meta, columns=feature_names)
    corr = df.corr().abs()
    top = corr.mean().sort_values(ascending=False).head(10).index.tolist()
    sub = df[top].corr()
    mask = np.triu(np.ones_like(sub, dtype=bool))
    plt.figure(figsize=(10,8))
    sns.heatmap(sub, mask=mask, cmap='coolwarm', annot=True, fmt='.2f')
    plt.title("Top 10 Feature Correlations")
    plt.savefig("feature_correlation_matrix.png")
    plt.close()


def save_feature_importance(model, X_seq_val, X_meta_val, y_val, feature_names, fold):
    base_acc = accuracy_score(y_val, (model.predict([X_seq_val, X_meta_val])>0.5).astype(int))
    imps = []
    for i in range(X_meta_val.shape[1]):
        perm = X_meta_val.copy()
        np.random.shuffle(perm[:, i])
        acc = accuracy_score(y_val, (model.predict([X_seq_val, perm])>0.5).astype(int))
        imps.append(base_acc - acc)
    idx = np.argsort(imps)
    vals = np.array(imps)[idx]
    names = np.array(feature_names)[idx]
    mask = vals > 0.005
    plt.figure(figsize=(10,6))
    plt.barh(np.arange(mask.sum()), vals[mask])
    plt.yticks(np.arange(mask.sum()), names[mask])
    plt.title(f"Top Feature Importances (Fold {fold})")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"feature_importance_fold_{fold}.png")
    plt.close()

# -- Main Execution ----------------------------------------------------------------

if __name__ == '__main__':
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    X_raw, y = load_json_data('data')
    X_seq, X_meta = preprocess_data(X_raw)
    y = np.array(y)

    scaler = StandardScaler()
    X_meta = scaler.fit_transform(X_meta)

    save_feature_correlations(X_meta, [
        'total_time','time_to_first_movement','total_pause_time','pause_count',
        'velocity_mean','velocity_std','velocity_min','velocity_max','velocity_skew',
        'acceleration_mean','acceleration_std','acceleration_min','acceleration_max','acceleration_skew',
        'curvature_mean','curvature_std','curvature_min','curvature_max','curvature_skew',
        'jerk_mean','jerk_std','jerk_min','jerk_max','jerk_skew'
    ])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq, y), start=1):
        print(f"\n--- Fold {fold} ---")
        X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
        X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(X_seq.shape[1:], X_meta.shape[1])
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(f'model_fold_{fold}.h5', save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]

        hist = model.fit(
            [X_seq_train, X_meta_train], y_train,
            validation_data=([X_seq_val, X_meta_val], y_val),
            epochs=30, batch_size=32,
            callbacks=callbacks, verbose=0
        )

        plot_training_history(hist, fold)

        y_val_prob = model.predict([X_seq_val, X_meta_val])
        y_val_pred = (y_val_prob > 0.5).astype(int)

        cm = save_confusion_matrix(y_val, y_val_pred, fold)
        save_roc_curve(y_val, y_val_prob, fold)
        save_precision_recall_curve(y_val, y_val_prob, fold)
        save_calibration_curve(y_val, y_val_prob, fold)
        save_feature_importance(model, X_seq_val, X_meta_val, y_val, [
            'total_time','time_to_first_movement','total_pause_time','pause_count',
            'velocity_mean','velocity_std','velocity_min','velocity_max','velocity_skew',
            'acceleration_mean','acceleration_std','acceleration_min','acceleration_max','acceleration_skew',
            'curvature_mean','curvature_std','curvature_min','curvature_max','curvature_skew',
            'jerk_mean','jerk_std','jerk_min','jerk_max','jerk_skew'
        ], fold)

        acc = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred, average='macro')
        print(f"Fold {fold} Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        results.append({'fold': fold, 'accuracy': acc, 'f1_macro': f1,
                        'tp': cm[1,1], 'fp': cm[0,1], 'tn': cm[0,0], 'fn': cm[1,0]})

    df_res = pd.DataFrame(results)
    df_res.loc['avg'] = df_res.mean(numeric_only=True)
    df_res.to_csv(f"cv_results_{run_id}.csv", index=False)
    print("\nAverage Results:")
    print(df_res.tail(1))


