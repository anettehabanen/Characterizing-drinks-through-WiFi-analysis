import torch.nn.functional
import keras_core as keras
import time
from threading import Thread
import subprocess
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"

data_history = []


def read_data():
    # command that runs ESP-IDF in powershell
    p = subprocess.Popen('C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe '
                         '-ExecutionPolicy Bypass -NoExit -Command "C:\\Espressif\\Initialize-Idf.ps1" '
                         '-IdfId esp-idf-7a3cc3545977e8fdecfa73521e1f3180; '
                         'cd C:\\Espressif\\ESP32-CSI-Tool-master\\ESP32-CSI-Tool-master\\active_sta; idf.py -p COM5 monitor',
                         stdout=subprocess.PIPE)

    cols = ("type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,"
            "noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,"
            "real_timestamp,len,CSI_DATA").split(",")

    for stdout_line in iter(p.stdout.readline, ""):
        try:
            l = stdout_line.decode("utf-8").strip()
            if "CSI_DATA," not in l:
                continue
            data = dict()
            for i in range(len(cols)):
                data[cols[i]] = l.split(",")[i]

            data["CSI_DATA"] = [int(x) for x in data["CSI_DATA"][1:-2].split(" ")]

            if len(data_history) >= 100:
                data_history.pop(0)

            data_history.append(data["CSI_DATA"])
        except:
            pass


if __name__ == '__main__':
    t = Thread(target=read_data)
    t.start()
    idx_to_label = {0: "water", 1: "yogurt", 2: "beer", 3: "plumJuice"}

    unneeded_cols = set(  # data cleaning
        list(range(0, 12)) +
        list(range(22, 24)) +
        list(range(50, 52)) +
        list(range(64, 66)) +
        list(range(78, 80)) +
        list(range(106, 108)) +
        list(range(118, 128))
    )

    # read in 10 different models
    models = [keras.saving.load_model(f"model{i}.h5") for i in range(1, 11)]

    models_per_class = {
        0: [3, 4],
        1: [2, 4, 8],
        2: [0, 1, 3, 4, 5, 6, 8, 9],
        3: [3, 9, 5, 6],
    }

    # gather enough data to start predictions
    while len(data_history) < 100:
        time.sleep(0.1)

    print("Starting predictions")

    while 1:  # prediction loop
        data = np.array(data_history)
        input_data = np.zeros(shape=(1, len(data_history), 96))  # [B, seq_len, features]
        j = 0
        for i in range(128):  # remove unneeded columns from data
            if i in unneeded_cols:
                continue
            input_data[0, :, j] = data[:, i]
            j += 1

        preds = [m.predict(input_data, verbose=0)[0] for m in models]

        this_probs = [0, 0, 0, 0]
        for k, v in models_per_class.items():
            this_probs[k] = sum(preds[model_idx][k] for model_idx in v)
            this_probs[k] /= len(v)

        pred = torch.nn.functional.softmax(torch.tensor(list(this_probs))).cpu().numpy()

        probs = "; ".join(f'{idx_to_label[i]}: {(pred[i] * 100):.02f}%' for i in range(len(pred)))
        print(f"\rPrediction: {idx_to_label[np.argmax(pred)]} \t\tProbabilities  {probs}", end="")
