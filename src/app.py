from flask import Flask, request, jsonify
from db.db import get_nearest_jobs, insert_jobs, delete_all_jobs, find_jobs_in_list_ids, find_queued_jobs, update_job
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from model.model import A2C
from model.model import (
    preprocessing_queued_jobs,
    preprocessing_system_status,
    make_feature_vector,
    get_action_from_output_vector,
)
import torch
import numpy as np
import joblib
from bson import ObjectId

app = Flask(__name__)

# Load model
window_size = 50
sys_size = 288
learning_rate = 0.000021
gamma = 0.95
batch_size = 70
layer_size = 4000, 1000
num_inputs = window_size * 3 + sys_size * 1
a2c = A2C(
    num_inputs,
    window_size,
    std=0.0,
    window_size=window_size,
    learning_rate=learning_rate,
    gamma=gamma,
    batch_size=batch_size,
    layer_size=layer_size,
)
a2c.load_using_model_name("model/agent")

# Load KNN model
current_max_id = 0
current_accuracy = 0
maxs = 0
mins = 0


@app.route("/hello")
def hello():
    return "Hello World!"


@app.route("/ai/schedule", methods=["POST"])
def schedule():
    raw_ids = request.get_json()["waiting_job_ids"]
    ids = [ObjectId(id) for id in raw_ids]
    job_info_dict_cursor = find_jobs_in_list_ids(ids)
    job_info_dict = {} # list infos of jobs in queue
    wait_que_indices = [] # list ids of jobs in queue
    for job in job_info_dict_cursor:
        print(job)
        job_info_dict[job["_id"]] = {
            "_id": job["_id"],
            "submissionTime": job["createdDate"].timestamp(),
            "userEst": job["userEst"],
            "procReq": job["procReq"],
            "userId": int(job["userId"], 16),
            "exe_num": job["exe_num"] if "exe_num" in job.keys() else int(job["userId"], 16),
        }
        # job_info_dict[job["_id"]] = job
        wait_que_indices.append(job["_id"])
    print(job_info_dict)
    print(wait_que_indices)
    # End of get infos of jobs in queue
    
    # Start schedule
    # Env
    current_time = request.get_json()["current_time"]
    wait_que_size = len(wait_que_indices)
    wait_job = [job_info_dict[ind] for ind in wait_que_indices]
    node_info_list = request.get_json()["node_info_list"]
    print(len(node_info_list))

    wait_job_input = preprocessing_queued_jobs(wait_job, current_time)
    system_status_input = preprocessing_system_status(node_info_list, current_time)
    feature_vector = make_feature_vector(wait_job_input, system_status_input)

    state = torch.FloatTensor(feature_vector)
    probs, value = a2c(state)
    action = get_action_from_output_vector(probs.detach(), wait_que_size, 0)
    if not can_allocate(wait_job[action], node_info_list):
        return jsonify({"status": 200, "job_id": None})
    return jsonify({"status": 200, "job_id": wait_que_indices[action].__str__()})

@app.route("/ai/v2/schedule", methods=["POST"])
def schedule_v2():
    job_info_dict_cursor = find_queued_jobs()
    job_info_dict = {} # list infos of jobs in queue
    wait_que_indices = [] # list ids of jobs in queue
    for job in job_info_dict_cursor:
        print(job)
        job_info_dict[job["_id"]] = {
            "_id": job["_id"],
            "submissionTime": job["createdDate"].timestamp(),
            "userEst": job["userEst"],
            "procReq": job["procReq"],
            "userId": int(job["userId"], 16),
            "exe_num": job["exe_num"] if "exe_num" in job.keys() else int(job["userId"], 16),
        }
        # job_info_dict[job["_id"]] = job
        wait_que_indices.append(job["_id"])
    print(job_info_dict)
    print(wait_que_indices)
    # End of get infos of jobs in queue
    
    # Start schedule
    # Env

    current_time = request.get_json()["current_time"]
    node_info_list = request.get_json()["node_info_list"]
    print("node info:", len(node_info_list))

    selected_job_ids = []
    selected_now = True
    while len(wait_que_indices) > 0:
        print("START wait_que_indices:", wait_que_indices)
        print("START wait_que_indices size:", len(wait_que_indices))
        print("START node_info_list:", sum(1 for i in node_info_list if i['end'] == 0))

        wait_que_size = len(wait_que_indices)
        wait_job = [job_info_dict[ind] for ind in wait_que_indices] # init wait job
        wait_job_input = preprocessing_queued_jobs(wait_job, current_time)
        system_status_input = preprocessing_system_status(node_info_list, current_time)
        feature_vector = make_feature_vector(wait_job_input, system_status_input)

        state = torch.FloatTensor(feature_vector)
        probs, value = a2c(state)
        action = get_action_from_output_vector(probs.detach(), wait_que_size, 0)
        #return jsonify({"status": 200, "job_id": wait_que_indices[action].__str__()})
        
        #update node after select job
        if not can_allocate(wait_job[action], node_info_list):
            selected_now = False
            for node in node_info_list:
                if node['end'] == 0:
                    continue
                current_time += node['end']*1000 # ms
                for node in node_info_list:
                    if node['end'] != 0:
                        node['end'] -= node['end']
                break
        else:
            update_job(wait_que_indices[action], {"predictStart": current_time,
                                                  "predictEnd": current_time + wait_job[action]['userEst']*1000})

            if selected_now:
                selected_job_ids.append(str(wait_que_indices[action]))
            wait_que_indices.pop(action)
            node_info_list = update_node_info_list(wait_job[action], node_info_list)

    return jsonify({"status": 200, "job_ids": selected_job_ids}) 

def can_allocate(job, node_info_list):
    num_nodes_free = sum(1 for i in node_info_list if i['end'] == 0)
    if num_nodes_free < job["procReq"]:
        return False
    return True

def update_node_info_list(job, node_info_list):
    num_proc_req = job["procReq"]
    for node in node_info_list:
        if node['end'] == 0:
            node['end'] = job['userEst']
            num_proc_req -= 1
            if num_proc_req == 0:
                break
    node_info_list.sort(key=lambda x: x['end'])
    return node_info_list


@app.route("/ai/soft_walltime", methods=["POST"])
def predict_soft_walltime():
    # Get lastest jobs
    history = get_nearest_jobs(1000)

    knn = None
    global current_max_id
    global current_accuracy
    global maxs
    global mins
    columns_to_scale = ["executionTime", "procReq"]
    
    # Select essential field
    data = history[["executionTime", "procReq", "userId"]]

    # Normalize
    maxs = data[columns_to_scale].max()
    mins = data[columns_to_scale].min()
    data[columns_to_scale] = (data[columns_to_scale] - mins) / (maxs - mins)

    # Devide x and y dataset
    x = data.drop(["executionTime"], axis=1)
    y = data["executionTime"]

    # Devide train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Train
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(x_train, y_train)

    joblib.dump(knn, "model/knn_model.joblib")

    # Test
    # predict test set
    y_pred = knn.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    current_accuracy = (((1 - mse) + r2) / 2) * 100
    

    # Get input
    job = request.get_json()
    # Rename job
    renamed_job = {
        "executionTime": job["userEst"],
        "procReq": job["procReq"],
        "userId": job["userId"],
    }
    # Covert to DataFrame
    job = pd.DataFrame([renamed_job])
    job[columns_to_scale] = (job[columns_to_scale] - mins) / (maxs - mins + 1e-6)

    y_pred_scaled = knn.predict(job.drop("executionTime", axis=1))
    y_pred = (
        y_pred_scaled[0] * (maxs["executionTime"] - mins["executionTime"])
        + mins["executionTime"]
    )

    result = {
        "status": 200,
        "walltime": y_pred,
        "accuracy": str(current_accuracy) + "%",
    }
    return result
    try:
        # Get lastest jobs
        nearest_job = get_nearest_jobs(1000)
        history = pd.DataFrame(nearest_job)
        history.drop("_id", axis=1, inplace=True)

        knn
        if current_max_id < history["jobID"].max():
            current_max_id = history["jobID"].max()
            # Select essential field
            data = history[["execution_time", "cpu_used", "uid", "gid"]]

            # Normalize
            columns_to_scale = ["execution_time", "cpu_used"]
            maxs = data[columns_to_scale].max()
            mins = data[columns_to_scale].min()
            data[columns_to_scale] = (data[columns_to_scale] - mins) / (maxs - mins)

            # Devide x and y dataset
            x = data.drop(["execution_time"], axis=1)
            y = data["execution_time"]

            # Devide train and test data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            # Train
            knn = KNeighborsRegressor(n_neighbors=5)
            knn.fit(x_train, y_train)

            joblib.dump(knn, "knn_model.joblib")

            # Test
            # predict test set
            y_pred = knn.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            current_accuracy = (((1 - mse) + r2) / 2) * 100
        else:
            knn = joblib.load("knn_model.joblib")

        # Get input
        job = request.get_json()
        # Rename job
        renamed_job = {
            "execution_time": job["user_est"],
            "cpu_used": job["proc_req"],
            "uid": job["uid"],
            "gid": job["gid"],
        }
        # Covert to DataFrame
        job = pd.DataFrame([renamed_job])
        job[columns_to_scale] = (job[columns_to_scale] - mins) / (maxs - mins + 1e-6)

        y_pred_scaled = knn.predict(job.drop("execution_time", axis=1))
        y_pred = (
            y_pred_scaled[0] * (maxs["execution_time"] - mins["execution_time"])
            + mins["execution_time"]
        )

        result = {
            "status": 200,
            "walltime": y_pred,
            "accuracy": str(current_accuracy) + "%",
        }
        return result
    except:
        return jsonify({"status": 500, "message": "Internal Server Error!"})


@app.route("/db/insert_jobs", methods=["POST"])
def insert_jobs_controller():
    # insert_jobs()
    return "Insert jobs successfully!"


@app.route("/db/delete_all_jobs", methods=["POST"])
def delete_all_jobs_controller():
    # delete_all_jobs()
    return "Delete jobs successfully!"


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0")
