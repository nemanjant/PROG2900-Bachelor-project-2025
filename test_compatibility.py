"""
Compatibility tests verifying all required packages work correctly together,
with focus on the TensorFlow 2.11 + NumPy 1.23.5 combination.
"""

import sys


def test_python_version():
    major, minor = sys.version_info[:2]
    assert (major, minor) == (3, 10), f"Expected Python 3.10, got {major}.{minor}"
    print(f"  Python {major}.{minor} OK")


def test_numpy_version():
    import numpy as np
    assert np.__version__ == "1.23.5", f"Expected numpy 1.23.5, got {np.__version__}"
    print(f"  NumPy {np.__version__} OK")


def test_numpy_type_aliases():
    """NumPy 1.23.x must still expose the aliases TF 2.11 relies on."""
    import numpy as np
    for alias in ("bool", "int", "float", "complex", "object", "str"):
        assert hasattr(np, alias), f"np.{alias} missing"
    print("  NumPy type aliases (bool/int/float/…) present OK")


def test_tensorflow_import():
    import tensorflow as tf
    assert tf.__version__ == "2.11.0", f"Expected tf 2.11.0, got {tf.__version__}"
    print(f"  TensorFlow {tf.__version__} OK")


def test_tensorflow_numpy_interop():
    """Core operation: TF tensor <-> NumPy array round-trip."""
    import numpy as np
    import tensorflow as tf

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tensor = tf.constant(arr)
    result = tensor.numpy()
    assert np.allclose(arr, result), "TF <-> NumPy round-trip mismatch"
    print("  TF <-> NumPy round-trip OK")


def test_tensorflow_keras_layers():
    """Verify the layer types used in the LSTM/GRU training scripts."""
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, LayerNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    import numpy as np

    inputs = Input(shape=(10, 6))
    x = LSTM(32, return_sequences=True)(inputs)
    x = GRU(16)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, x)

    dummy = np.zeros((2, 10, 6), dtype=np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (2, 1), f"Unexpected output shape: {out.shape}"
    print("  Keras LSTM + GRU model forward-pass OK")


def test_sklearn():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
    import numpy as np

    X = np.random.rand(40, 5).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_scaled, y)
    preds = clf.predict(X_scaled)
    acc = accuracy_score(y, preds)
    assert 0.0 <= acc <= 1.0
    print(f"  scikit-learn RandomForest OK (train acc={acc:.2f})")


def test_pandas():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({"x": np.arange(5), "y": np.arange(5) * 2.0})
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 5
    print("  pandas DataFrame OK")


def test_scipy():
    from scipy.stats import skew
    import numpy as np

    data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    s = skew(data)
    assert s > 0, "Expected positive skew"
    print(f"  scipy.stats.skew OK (skew={s:.2f})")


def test_matplotlib_seaborn():
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe in all environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fig, ax = plt.subplots()
    sns.histplot(np.random.randn(100), ax=ax)
    plt.close(fig)
    print("  matplotlib + seaborn plot OK")


def test_jsonschema():
    import jsonschema

    schema = {
        "type": "object",
        "properties": {"mouseMovements": {"type": "array"}},
        "required": ["mouseMovements"],
    }
    jsonschema.validate({"mouseMovements": [[0, 0], [1, 1]]}, schema)
    print("  jsonschema validate OK")


def run_all():
    tests = [
        test_python_version,
        test_numpy_version,
        test_numpy_type_aliases,
        test_tensorflow_import,
        test_tensorflow_numpy_interop,
        test_tensorflow_keras_layers,
        test_sklearn,
        test_pandas,
        test_scipy,
        test_matplotlib_seaborn,
        test_jsonschema,
    ]

    passed, failed = 0, []
    for t in tests:
        name = t.__name__
        try:
            print(f"[RUN] {name}")
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print("Failed:", ", ".join(failed))
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    run_all()
