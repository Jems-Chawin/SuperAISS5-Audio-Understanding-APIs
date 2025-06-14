# 📝 Context  
ไฟล์นี้คือ instruction สำหรับ code ที่ผมนำมา knowledge sharing 
สามารถอ่านและทำตามขั้นตอนได้เลยครับ บางจุดอาจมีข้อผิดพลาดหรือเกิด error ได้แม้จะทำตามครบทุกขั้นตอน ถือว่าเป็นเรื่องปกติ  
หากติดปัญหา สามารถถามแชทช่วยได้เลยครับ 😊  

---

# ✅ Steps

## 0. ดาวน์โหลดโมเดลจาก Hugging Face  
ใช้คำสั่งนี้เพื่อลงโมเดล:  
```bash
wget https://huggingface.co/Jemssss/Three_whisper_model/resolve/main/best_model_2trans.pt
```

---

## 1. เปิดอ่าน Markdown ใน VS Code Preview  
สำหรับคนที่ใช้ **VS Code**  
กด `Ctrl + K` แล้วตามด้วย `V` เพื่อเปิดโหมด **Markdown Preview**  
จะช่วยให้อ่านไฟล์นี้ได้ง่ายขึ้นครับ  

---

## 2. ติดตั้ง Virtual Environment

> หมายเหตุ: คำสั่งเหล่านี้เป็นสำหรับ **Linux/Mac**  
> ถ้าใช้ **Windows** ให้ copy ไปถามแชทเพื่อแปลงให้เหมาะกับระบบของคุณครับ  

### 🔹 สำหรับ `API`

1. สร้าง venv:  
   ```bash
   python3 -m venv venv_api
   ```

2. เปิดใช้งาน venv:  
   ```bash
   source venv_api/bin/activate
   ```

3. ติดตั้ง dependencies:  
   ```bash
   pip install -r requirements_api.txt
   ```
   > ถ้าเจอ bug หรือ error ระหว่างทาง ให้แชทช่วยได้เลยครับ

---

### 🔹 สำหรับ `Loadtest`

1. สร้าง venv:  
   ```bash
   python3 -m venv venv_loadtest
   ```

2. เปิดใช้งาน venv:  
   ```bash
   source venv_loadtest/bin/activate
   ```

3. ติดตั้ง dependencies:  
   ```bash
   pip install -r requirements_loadtest.txt
   ```

เมื่อทำครบแล้ว ถือว่าเสร็จสิ้นการตั้งค่า environment ครับ 🎉

---

## 3. รันไฟล์ `api_multimodal.py`

ไฟล์นี้คือ **API หลัก** ที่ใช้ส่ง blindtest  
ขั้นตอนมีดังนี้:

1. เปิดใช้งาน venv:
   ```bash
   source venv_api/bin/activate
   ```

2. รัน API:
   ```bash
   python api_multimodal.py
   ```

ถ้าเปิดสำเร็จ เข้าไปที่:  
[http://127.0.0.1:4000/docs](http://127.0.0.1:4000/docs)  
ถ้าเจอหน้าต่าง **FastAPI UI** แปลว่าใช้ได้แล้วครับ ✅

---

## 4. รันไฟล์ `loadtest.py`

ใช้สำหรับทำ **load testing** พร้อมเก็บคะแนน performance

1. เปิด Terminal ใหม่ (ปล่อย terminal เดิมที่รัน API ทิ้งไว้)

2. เปิดใช้งาน venv สำหรับ loadtest:
   ```bash
   source venv_loadtest/bin/activate
   ```

3. รัน loadtest:
   ```bash
   locust -f loadtest.py --users 10 --spawn-rate 1 --run-time 10m --host http://0.0.0.0:4000 --web-port 9000
   ```

4. เข้า UI ของ Locust ได้ที่:  
   [http://127.0.0.1:9000](http://127.0.0.1:9000) แล้วกด **Start**

> Locust จะยิง request ไปยัง API จนกว่าจะครบเวลา (10 นาที)

---

## 5. ไฟล์ `process_data.py`

เป็นไฟล์ที่ใช้สำหรับ **แปลงข้อมูลให้อยู่ในรูปแบบที่ `loadtest` ต้องการ**  
ถ้าอยากใช้ข้อมูลของตัวเองยิงทดสอบ อาจต้องดู logic ในไฟล์นี้เพื่อปรับ format ให้ตรงกันครับ

---

# 📁 โฟลเดอร์อื่น ๆ ที่ควรรู้จัก

### ▪️ `utils/`  
เก็บโมดูลที่ใช้งานร่วมกับ loadtest  
สามารถเข้าไปดูโค้ดหรือปรับแต่งเพิ่มเติมได้  

### ▪️ `Three_models/`  
เก็บไฟล์ `.pt` ซึ่งเป็นโมเดลของทรีครับ  

### ▪️ `data_handmade/`  
เก็บ data ที่ทีม data เตรียมไว้ให้  
ใช้สำหรับยิง API ผ่าน loadtest  

### ▪️ `temp/`  
เก็บไฟล์ API อื่น ๆ ที่เคยทดลอง  
อาจรันไม่ได้ เพราะไม่มี venv สำหรับพวกนี้  
หากสนใจสามารถลองแกะโค้ดและสร้าง venv เองครับ
