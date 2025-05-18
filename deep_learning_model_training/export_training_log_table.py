import pandas as pd
import matplotlib.pyplot as plt
import os

def save_training_log_table_as_png(csv_path, output_path='training_table.png', title=None):
    
    # Read training log CSV
    df = pd.read_csv(csv_path)

    expected_columns = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"CSV must contain columns: {expected_columns}")

    # Add epoch column (if not already present)
    df.insert(0, 'Epoch', range(1, len(df) + 1))

    # Round values for cleaner display
    df_rounded = df.round(4)

    # Create figure
    fig_height = len(df) * 0.3 + (1.5 if title else 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')

    # Draw table
    table = ax.table(
        cellText=df_rounded.values,
        colLabels=df_rounded.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Add title
    if title:
        plt.title(title, fontsize=14, pad=20)

    # Save as PNG
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training log table with title to {output_path}")

# Example usage
if __name__ == "__main__":
    csv_file = 'training_log_fold_5.csv'  # Replace with your path
    output_file = 'fold5_training_table.png'
    table_title = 'Training Log (Fold 5)'
    save_training_log_table_as_png(csv_file, output_file, title=table_title)

