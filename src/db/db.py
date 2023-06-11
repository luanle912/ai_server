from pymongo import MongoClient
from flask import jsonify
import pandas as pd

client = MongoClient('mongodb+srv://lvtn:lvtn123lvtn@cluster0.f5i6j.mongodb.net/lvtn?retryWrites=true&w=majority')
db = client['lvtn']

def get_nearest_jobs(num_jobs=1000):
    # Query 1000 nearest jobs from db
    collection = db['jobs']
    
    cursor = collection.find({'status': 2}).sort('submissionTime', -1).limit(num_jobs)

    cursor = pd.DataFrame(cursor)
    columns_to_keep = ['executionTime', 'procReq', 'userId']
    cursor = cursor.drop(columns=[col for col in cursor.columns if col not in columns_to_keep], axis = 1)
    print('cursor_list 1', cursor.head())

    new_row = get_job_in_swf_file()
    print('new_row', new_row.head())

    merged_df = pd.concat([cursor, new_row[:num_jobs - cursor.shape[0]]])

    # Convert the list back to a cursor
    merged_df['userId'] = merged_df['userId'].factorize()[0]
    merged_df['executionTime'] = merged_df['executionTime'].astype(int)
    merged_df['procReq'] = merged_df['procReq'].astype(int)
    print('cursor_list 2', merged_df.head())
    return merged_df

def get_job_in_swf_file(input_file_path='data/workloads.swf'):
    # Đường dẫn đến tệp SWF
    file_path = input_file_path

    # Đọc từng dòng trong tệp SWF và chuyển đổi thành DataFrame
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Loại bỏ ký tự newline và khoảng trắng thừa
            if line and not line.startswith(';'):  # Bỏ qua dòng trống và dòng bắt đầu bằng ';'
                row = line.split('\t')  # Phân chia dữ liệu theo ký tự tab
                data.append(row)

    df = pd.DataFrame(data)
    df = df.iloc[:, [3, 7, 11]]
    df = df.rename(columns={3: 'executionTime', 7: 'procReq', 11: 'userId'})
    df = df.dropna()
    return df

def get_job_by_id(id):
    # Query job by id from db
    pass

def insert_jobs(file_path='data/workloads.swf'):
    # Insert jobs from file to db
    collection = db['jobs']
    columns = ['jobID', 'submission_time', 'waiting_time',
                   'execution_time', 'proc_alloc', 'cpu_used', 'mem_used',
                   'proc_req', 'user_est', 'mem_req', 'status', 'uid',
                   'gid', 'exe_num', 'queue', 'partition', 'prev_jobs',
                   'think_time']
    data = pd.read_csv(file_path, comment=';', names=columns, delim_whitespace=True)
    print(data.head())
    # Chuyển đổi dữ liệu thành danh sách từ điển
    records = data.to_dict("records")

    # Thêm từng bản ghi vào collection trong MongoDB
    for record in records:
        print(record)
        collection.insert_one(record)

def delete_all_jobs():
    collection = db['jobs']
    result = collection.delete_many({})
    print(result.deleted_count, "documents deleted.")
    return "Documents deleted!"

def clean_jobs():
    collection = db['jobs']
    result = collection.delete_many({"mem_used": {"$exists": True}})

    # In số tài liệu đã bị xóa
    print(f"Số tài liệu đã bị xóa: {result.deleted_count}")

    # Đóng kết nối tới MongoDB
    client.close()

def find_jobs_in_list_ids(ids):
    collection = db['jobs']
    result = collection.find({'_id': {'$in': ids}})
    return result

def find_queued_jobs():
    collection = db['jobs']
    result = collection.find({'status': 0})
    return result

def update_job(document_id, update_dict):
    collection = db['jobs']
    # Specify the filter criteria to find the document to update
    filter_criteria = {'_id': document_id}

    # Specify the update operation
    update_operation = {'$set': update_dict}

    # Update the document
    result = collection.update_one(filter_criteria, update_operation)

    # Check if the update was successful
    if result.modified_count > 0:
        print("Field updated successfully.")
    else:
        print("No document matching the criteria was found.")

