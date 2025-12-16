import time
import psutil
import numpy as np

def measure_training_resources(train_function):
    cpu_usage = []
    ram_usage = []

    start = time.time()

    result = train_function(
        cpu_usage_list=cpu_usage,
        ram_usage_list=ram_usage
    )

    end = time.time()

    return {
        "training_time_sec": end - start,
        "cpu_avg": np.mean(cpu_usage),
        "ram_avg": np.mean(ram_usage),
        "result": result
    }


def monitor_prediction(predict_function):
    start = time.time()
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent

    result = predict_function()

    end = time.time()

    return {
        "prediction_time_sec": end - start,
        "cpu": cpu,
        "ram": ram,
        "result": result
    }
