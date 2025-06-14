# process_data.py
# python process_data.py

''' this file is for preprocessing the columns of the train.csv '''

import pandas as pd

# 1) Load your original CSV

# file_name = "/home/siamai/data/datasets/Audio Understanding SCBx/speechs/train"
file_name = "./data_handmade/trainData/train"
df = pd.read_csv(file_name + ".csv")

# only for data handmade
if file_name == "./data_handmade/trainData/train":
    df["agent_fname"] = "John"
    df["agent_lname"] = "Doe"

# 2) Rename columns to match TestDataRow fields:
df = df.rename(columns={
    "id":               "voice_file_path",
    # "first_name":       "agent_fname",              # if first_name is actually your agent’s name
    # "last_name":        "agent_lname",              # likewise for last_name
    "กล่าวสวัสดี":       "is_greeting",
    "แนะนำชื่อและนามสกุล": "is_introself",
    "บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ":
                        "is_informlicense",
    "บอกวัตถุประสงค์ของการเข้าพบครั้งนี้":
                        "is_informobjective",
    "เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้":
                        "is_informbenefit",
    "บอกระยะเวลาที่ใช้ในการเข้าพบ":
                        "is_informinterval",
})

# 3) If you have an “expected” first/last name for the voice, also rename those
#    (otherwise they'll be left out or filled with dummy values)
# df["voice_fname"] = df["agent_fname"]
# df["voice_lname"] = df["agent_lname"]
df = df.loc[:, ['voice_file_path', 'agent_fname', 'agent_lname', 'is_greeting', 'is_introself', 'is_informlicense', 'is_informobjective', 'is_informbenefit', 'is_informinterval']]

# 4) Save back (you can overwrite or to a new file)
print(df.head())
df.to_csv(f"processed_handmade.csv", index=False)
# df.to_csv(f"processed_{file_name}.csv", index=False)
print("✅ Wrote processed_train.csv with", df.shape[0], "rows and columns:", list(df.columns))