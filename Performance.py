import time
import psutil
import numpy as np
from memory_profiler import memory_usage

def measure_performance(train_func, predict_func, X_test, repeat=1):

    # 1) Training performance

    def _train_wrapper():
        return train_func()

    mem_train = memory_usage((_train_wrapper,), max_iterations=1, interval=0.1)
    mem_train_peak = max(mem_train)

    start_train = time.time()
    model = train_func()
    end_train = time.time()

    cpu_train = psutil.cpu_percent(interval=None)

    
    # 2) Prediction performance
    
    def _predict_wrapper():
        return predict_func(model, X_test)

    mem_pred = memory_usage((_predict_wrapper,), max_iterations=1, interval=0.1)
    mem_pred_peak = max(mem_pred)

    start_pred = time.time()
    _ = predict_func(model, X_test)
    end_pred = time.time()

    cpu_pred = psutil.cpu_percent(interval=None)


    # Return results 
   
    return {
        "train_time_sec": end_train - start_train,
        "train_cpu_percent": cpu_train,
        "train_ram_peak_MB": mem_train_peak,

        "predict_time_sec": end_pred - start_pred,
        "predict_cpu_percent": cpu_pred,
        "predict_ram_peak_MB": mem_pred_peak,
    }
