# Context
นี่คือไฟล์ markdown ที่ผมทำเป็น instruction ไฟล์เอาไว้สำหรับ อ่านเพื่อทำตามขั้นตอนได้เลยครับ แต่อาจจะมีบางจุดตกหล่น หรือเกิด error ได้ต่อให้ทำตามแล้ว เป็นเรื่องปกติ ให้แชทช่วยก็ได้ครับ
# Steps
## 0. Download model ได้จาก hugginface โดยใช้คำสั่ง 
`wget https://huggingface.co/Jemssss/Three_whisper_model/resolve/main/best_model_2trans.pt`
## 1. เปิดอ่านในโหมด preview
ถ้าใครที่เปิดอ่านไฟล์นี้ใน vscode จะสามารถอ่านได้ง่ายขึ้นในโหมด preview นะครับ โดยการกด `ctrl + k` แล้วกด `v` จะขึ้นแสดงหน้าต่าง preview ขึ้นมาครับ
## 2. ติดตั้ง virtual environment
อันดับแรกเลย เราก็จะต้องติดตั้ง env ที่ผมใช้ตอนรัน api ให้ครบก่อนครับ โดยคำสั่งที่ผมให้ไปนี้จะเป็นสำหรับ Linux/Mac นะครับ เพราะบน vm เป็นแบบนั้น สำหรับ windows ให้ลอง copy ทั้งหมดนี้โยนให้แชทมันนำทางให้นะครับ \
### 1. ใช้คำสั่ง `python3 -m venv venv_api` จะได้โฟลเดอร์ venv สำหรับตัว api มา
### 2. ใช้คำสั่ง `source venv_api/bin/activate` จะทำการ activate venv_api ขึ้นมา (สังเกตุได้จากข้างหน้าสุดของ cmd line terminal)
### 3. ใช้คำสั่ง `pip install -r requirements_api.txt` จะทำการลง dependencies ที่ผมใช้ทั้งหมด (กรณีที่มันเกิด bug หรือ error อะไรสักอย่างก็คงต้องให้แชทช่วยเหมือนเดิมครับ)
เป็นอันเสร็จสิ้นในส่วนของการลง venv สำหรับรัน api ครับ ต่อไปก็จะเป็นในส่วนของ venv สำหรับ loadtest ครับ คือ
### 1. ใช้คำสั่ง `python3 -m venv venv_loadtest` จะได้โฟลเดอร์ venv สำหรับ loadtest มา
### 2. ใช้คำสั่ง `source venv_loadtest/bin/activate` จะทำการ activate venv_loadtest ขึ้นมา (สังเกตุได้จากข้างหน้าสุดของ cmd line terminal)
### 3. ใช้คำสั่ง `pip install -r requirements_loadtest.txt` จะทำการลง dependencies ทั้งหมดสำหรับส่วนของ loadtest (เช่นเดิมนะครับ อาจจะมี bug หรือ error เกิดขึ้นได้ระหว่างทาง ให้น้องแชทช่วยเลยครับ)
เป็นอันเสร็จสิ้นการลง venv ทั้งหมดที่ใช้ครับ
## 3. run ไฟล์ api_multimodal.py
ไฟล์นี้ คือ ไฟล์หลักที่ใช้ในการส่ง blindtest ครับ โดยจะเป็นโมเดลที่ทรีทำ มีขั้นตอนที่ต้องทำ ดังนี้ครับ
### 1. เรียกใช้ venv_api ด้วยคำสั่ง `source venv_api/bin/activate` จะเป็นการ activate venv เพื่อรันไฟล์ครับ
### 2. ใช้คำสั่ง `python api_multimodal.py` จะทำการเปิดการทำงานของ api ขึ้นมาครับ
เป็นอันเสร็จสิ้นการรัน api ให้ทำงานไว้รอโดนยิงครับ สามารถเข้าผ่านทางหน้าเว็บไซต์ได้โดยการพิมพ์ http://127.0.0.1:4000/docs ถ้าขึ้นเป็นหน้าต่าง FastAPI UI ขึ้นมาแสดงว่าใช้ได้แล้วครับ
## 4. run ไฟล์ loadtest.py
ไฟล์นี้ คือ ไฟล์ที่ใช้ในการรัน loadtest แบบมีการวัดคะแนนต่างๆประกอบไปด้วย มีขั้นตอนดังนี้ครับ
### 1. กดเปิด terminal อีกหน้านึงขึ้นมาครับ (ปล่อย terminal ที่รัน api ไว้ครับ)
### 2. ใช้คำสั่ง `source venv_loadtest/bin/activate` เพื่อเป็นการ activate venv เพื่อรันไฟล์ครับ
### 3. ใช้คำสั่ง `locust -f loadtest.py --users 10 --spawn-rate 1 --run-time 10m --host http://0.0.0.0:4000 --web-port 9000` จะทำการรัน process ของตัว loadtest locust ขึ้นมาครับ
### 4. เปิดหน้าต่าง locust UI ขึ้นมาด้วย http://127.0.0.1:9000 แล้วกดปุ่ม start ครับ
เสร็จแล้ว locust จะทำการยิง request ไปยัง api ของเราเรื่อยๆจนกว่าจะครบทุก request หรือหมดเวลาที่กำหนดครับ (10 นาที)
## 5. ไฟล์ process_data.py
ไฟล์นี้เป็นไฟล์เสริมที่ผมใช้ในการ process data ให้เป็นรูปแบบที่ผมเขียนโค้ดเอาไว้ใน loadtest ครับ หากใครอยากจะลองใช้ data ของตัวเองในการยิงดูก็อาจจะต้องทำการเปลี่ยน format ให้เป็นไปตามที่ไฟล์นี้ทำครับ
# Others
* ## Folder utils:
เป็น folder ที่ใช้เก็บ module อื่นๆที่มีส่วนใช้งานในตัว loadtest ครับ สามารถไปแกะดูได้
* ## Folder Three_models:
เป็น folder เก็บไฟล์ .pt ตัวโมเดลของทรีครับ
* ## Folder data_handmade:
เป็น folder เก็บไฟล์ data ที่ฝั่งทีม data เป็นคนทำครับ ใช้ในการนำมายิง api ผ่าน loadtest
* ## Folder temp: 
เป็น folder เก็บไฟล์ api ท่าอื่นๆครับ เผื่ออยากลองเอาไปแกะกันดู แต่คงจะรันไม่ได้ครับ เพราะผมไม่ได้เก็บ venv สำหรับไฟล์พวกนี้ไว้ อาจจะต้องลองแกะแล้วสร้าง venv ขึ้นมาเองดูครับ
