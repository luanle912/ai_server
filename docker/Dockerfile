# Bước 1: Sử dụng image có sẵn của Python với phiên bản 3.9.6
FROM python:3.9.6

# Bước 2: Sao chép tệp requirements.txt vào container
COPY ./docker/requirements.txt .

# Bước 3: Cài đặt các thư viện được liệt kê trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bước 4: Đặt thư mục làm việc mặc định cho container
WORKDIR /app

# Bước 5: Sao chép tất cả các tệp trong thư mục hiện tại vào container tại thư mục /app
COPY . .

# Bước 6: Chạy ứng dụng Python
CMD ["python", "src/app.py"]
