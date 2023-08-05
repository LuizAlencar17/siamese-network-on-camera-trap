import json
import pandas as pd
import humanfriendly

from tqdm import tqdm
from absl import flags
from sklearn.metrics import classification_report, confusion_matrix

FLAGS = flags.FLAGS

def get_model_path():
    path = f'{FLAGS.checkpoint_path}{FLAGS.model_name}_{FLAGS.dataset_name}_{FLAGS.tag}_{FLAGS.image_size}/'
    print('model:', path)
    return path

def get_model_serengeti():
    return f'{FLAGS.checkpoint_path}{FLAGS.model_name}_serengeti_(tag:no_serengeti_weights)_{FLAGS.image_size}/'

def get_model_path_test_results():
    return f'test_{FLAGS.model_name}_{FLAGS.dataset_name}_{FLAGS.tag}_{FLAGS.image_size}'

def get_history_name():
    return f"{FLAGS.model_name}_{FLAGS.dataset_name}_{FLAGS.tag}_{FLAGS.image_size}"

def process_predicts(y_pred, dataset, tag):
    y_real_bool, y_pred_bool, y_pred_conf = [], [], []
    file_name_x, file_name_y = [], []
    idx_pred = 0
    for batch in tqdm(dataset):
        idx_file = 0
        file_names_batch, labels_batch = batch[0], batch[1]
        for value in labels_batch:
            y_real_bool.append(int(value[0]))
            y_pred_bool.append(round(y_pred[idx_pred][0]))
            y_pred_conf.append(y_pred[idx_pred][0])
            if len(file_names_batch) == 2:
                file_name_x.append(file_names_batch[0][idx_file].numpy().decode())
                file_name_y.append(file_names_batch[1][idx_file].numpy().decode())
            else:
                file_name_x.append(file_names_batch[idx_file].numpy().decode())
            idx_pred += 1
            idx_file += 1
    data = {
        "pred_confidence": y_pred_conf,
        "pred": y_pred_bool,
        "real": y_real_bool,
        "file_name_x": file_name_x}
    if file_name_y:
        data["file_name_y"] = file_name_y
    pd.DataFrame(data).to_csv(f'tmp/test_data/eval_{tag}.csv',
                              index=False)
    print(classification_report(y_real_bool, y_pred_bool))
    print(confusion_matrix(y_real_bool, y_pred_bool))

def save_history(history, title, epochs=10):
    df = pd.DataFrame(history)
    ax = df[['accuracy', 'val_accuracy']].plot(figsize=(10, 5))
    value = df['accuracy'].values[-1:][0]
    ax.annotate(str(round(value, 4)), xy=(epochs - 1, value))
    value = df['val_accuracy'].values[-1:][0]
    ax.annotate(str(round(value, 4)), xy=(epochs - 1, value))
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epocas")
    ax.get_figure().savefig(f"./tmp/train_data/{title}-accuracy.png")

    ax = df[['loss', 'val_loss']].plot(figsize=(10, 5))
    value = df['loss'].values[-1:][0]
    ax.annotate(str(round(value, 4)), xy=(epochs - 1, value))
    value = df['val_loss'].values[-1:][0]
    ax.annotate(str(round(value, 4)), xy=(epochs - 1, value))
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epocas")
    ax.get_figure().savefig(f"./tmp/train_data/{title}-loss.png")

def save_inference_time(model, images, elapsed):
    with open(f'tmp/test_time/{model}.json', 'w') as fp:
        json.dump({
            'model': model,
            'images': images,
            'time': humanfriendly.format_timespan(elapsed)
        }, fp)