
# Di chuyển vào thư mục dự án
cd E:\TNu\tnu-aiqa-kbqa

# Tạo venv
python -m venv venv

# Kích hoạt (Windows CMD)
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload