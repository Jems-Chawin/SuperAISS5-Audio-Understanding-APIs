'''
data train.csv
================================================================================
🎯 MULTIMODAL LLM LOAD TEST RESULTS
================================================================================
📊 Request Performance:
   Total Requests: 100
   Successful: 100
   Failed: 0
   Timeouts: 0
   Success Rate: 100.00%
   Failure Rate: 0.00%
   Timeout Rate: 0.00%

🎯 Accuracy Metrics:
   Greeting Detection: 98.00%
   Self Introduction: 98.00%
   License Information: 100.00%
   Objective Information: 93.00%
   Benefit Information: 96.00%
   Interval Information: 100.00%

📈 F₂ Score Breakdown:
   Greeting F₂: 98.11%
   Self Introduction F₂: 91.84%
   License Information F₂: 100.00%
   Objective Information F₂: 96.08%
   Benefit Information F₂: 96.85%
   Interval Information F₂: 100.00%

📊 Detailed Metrics (where F₂ ≠ Accuracy):

   Greeting:
      Precision: 98.11%
      Recall: 98.11%
      True Positives: 52
      False Positives: 1
      False Negatives: 1

   Intro Self:
      Precision: 100.00%
      Recall: 90.00%
      True Positives: 18
      False Positives: 0
      False Negatives: 2

   Inform License:
      Precision: 100.00%
      Recall: 100.00%
      True Positives: 52
      False Positives: 0
      False Negatives: 0

   Inform Objective:
      Precision: 89.09%
      Recall: 98.00%
      True Positives: 49
      False Positives: 6
      False Negatives: 1

   Inform Benefit:
      Precision: 93.48%
      Recall: 97.73%
      True Positives: 43
      False Positives: 3
      False Negatives: 1

   Inform Interval:
      Precision: 100.00%
      Recall: 100.00%
      True Positives: 57
      False Positives: 0
      False Negatives: 0

🏆 Overall Performance:
   Overall Accuracy: 97.50%
   Average F₂ Score: 97.72%

💾 Detailed results saved to: load_test_results/multimodal_loadtest_results_20250529_220153.json
================================================================================
'''


import os
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperModel
import torch.nn as nn
import tempfile
import json
from fastapi.middleware.cors import CORSMiddleware

# ================ CONFIG =================
MODEL_NAME = "nectec/Pathumma-whisper-th-large-v3"
MODEL_PATH = "./Three_models/best_model_2trans.pt"
NUM_LABELS = 6
SAMPLING_RATE = 16000
THRESHOLD = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================ LOAD MODEL COMPONENTS =================
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
whisper_model = WhisperModel.from_pretrained(MODEL_NAME)

class WhisperClassifier(nn.Module):
    def __init__(self, whisper_model=whisper_model, num_labels=NUM_LABELS):
        super().__init__()
        self.encoder = whisper_model.encoder
        self.encoder_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1280, nhead=8, dropout=0.1, batch_first=True, activation='gelu'),
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_labels),
        )
        self.weight_proj = nn.Linear(1280, 1)
    def forward(self, input_features_1, input_features_2):
        outputs_1 = self.encoder(input_features=input_features_1).last_hidden_state
        outputs_2 = self.encoder(input_features=input_features_2).last_hidden_state
        cat_outputs = torch.cat([outputs_1, outputs_2], dim=1)
        x_attn = self.encoder_block(cat_outputs)
        weights = torch.softmax(self.weight_proj(x_attn), dim=1)
        pooled = (x_attn * weights).sum(dim=1)
        logits = self.classifier(pooled)
        return logits

# -------------------- Load model --------------------
model = WhisperClassifier(num_labels=NUM_LABELS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

LABEL_COLUMNS = [
    'กล่าวสวัสดี',
    'แนะนำชื่อและนามสกุล',
    'บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ',
    'บอกวัตถุประสงค์ของการเข้าพบครั้งนี้',
    'เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้',
    'บอกระยะเวลาที่ใช้ในการเข้าพบ'
]


# ================ RESPONSE FORMAT ================
class AudioAnalysisResponse(BaseModel):
    transcription: str
    is_greeting: bool
    is_introself: bool
    is_informlicense: bool
    is_informobjective: bool
    is_informbenefit: bool
    is_informinterval: bool


# ================ FASTAPI INIT ================
app = FastAPI(
    title="WhisperClassifier-compatible ASR Eval API",
    description="Mimics Ray Serve /eval endpoint for WhisperClassifier model.",
    version="1.0.0"
)

# ---------- For frontend --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Preprocess inputs --------------------
def preprocess_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        f.flush()
        audio_path = f.name
    waveform, sr = torchaudio.load(audio_path)
    os.unlink(audio_path)
    if sr != SAMPLING_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLING_RATE)(waveform)
    waveform = waveform.squeeze(0)
    if waveform.dim() == 2:
        waveform = torch.mean(waveform, dim=0)
    chunk_lst = []
    for num_chuck in range(2):
        inputs = processor(waveform[num_chuck * 480000:(num_chuck+1) * 480000], sampling_rate=SAMPLING_RATE, return_tensors="pt")
        input_features = inputs.input_features.squeeze(0)
        chunk_lst.append(input_features)
    return chunk_lst


# ================ POST Method - evaluation ================
@app.post("/eval", response_model=AudioAnalysisResponse)
async def eval_endpoint(
    voice_file: UploadFile = File(...),
    agent_data: str = Form(...)  # not used, just for compatibility
):
    # Step 1: Validate and load audio file
    if not voice_file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    audio_bytes = await voice_file.read()
    try:
        chunk_lst = preprocess_audio(audio_bytes)
        input_features_1 = chunk_lst[0].unsqueeze(0).to(DEVICE)
        input_features_2 = chunk_lst[1].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_features_1, input_features_2)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > THRESHOLD).astype(bool).tolist()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    # Step 2: (Optional) Use Whisper decoder to transcribe (for "transcription" field)
    # Here, just a simple ASR decode (not using LLM)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            f.flush()
            audio_path = f.name
        input_asr = processor(audiopath=audio_path, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        input_features = input_asr.input_features.to(DEVICE)
        with torch.no_grad():
            # NOTE: WhisperModel does not include a decoder for speech-to-text, only encoder
            # You might want to use WhisperForConditionalGeneration or another ASR pipeline
            transcription = "<not_implemented>"
        os.unlink(audio_path)
    except Exception as e:
        transcription = "<transcription_error>"

    # Step 3: Format results in Ray Serve style
    return AudioAnalysisResponse(
        transcription="",  # You can return blank or fake if needed
        is_greeting=bool(preds[0]),
        is_introself=bool(preds[1]),
        is_informlicense=bool(preds[2]),
        is_informobjective=bool(preds[3]),
        is_informbenefit=bool(preds[4]),
        is_informinterval=bool(preds[5]),
    )


# ================ GET Method - simple root check ================
@app.get("/")
async def root():
    return {"msg": "WhisperClassifier API is running. POST audio to /eval"}


# ================ MAIN ================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=4000,
        workers=1,
        log_level="info",
    )