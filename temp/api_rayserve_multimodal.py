import os
import torch
import torchaudio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperModel
import torch.nn as nn
import tempfile

import ray
from ray import serve

# ======== CONFIG =========
MODEL_NAME = "nectec/Pathumma-whisper-th-large-v3"
MODEL_PATH = "/home/siamai/data/models/Three_models/best_model_2trans.pt"
NUM_LABELS = 6
SAMPLING_RATE = 16000
THRESHOLD = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLUMNS = [
    'กล่าวสวัสดี',
    'แนะนำชื่อและนามสกุล',
    'บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ',
    'บอกวัตถุประสงค์ของการเข้าพบครั้งนี้',
    'เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้',
    'บอกระยะเวลาที่ใช้ในการเข้าพบ'
]

class AudioAnalysisResponse(BaseModel):
    transcription: str
    is_greeting: bool
    is_introself: bool
    is_informlicense: bool
    is_informobjective: bool
    is_informbenefit: bool
    is_informinterval: bool

# ---------------------
# PyTorch Model
# ---------------------
class WhisperClassifier(nn.Module):
    def __init__(self, whisper_model, num_labels):
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

# ========== Ray Serve Deployment ==========

app = FastAPI()

@serve.deployment(ray_actor_options={"num_gpus": 1, "num_cpus": 2})
@serve.ingress(app)
class WhisperClassifierAPI:
    def __init__(self):
        # Load model and processor ONCE per replica
        self.processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        whisper_model = WhisperModel.from_pretrained(MODEL_NAME)
        self.model = WhisperClassifier(whisper_model, num_labels=NUM_LABELS).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        self.model.eval()
        self.sampling_rate = SAMPLING_RATE

    def preprocess_audio(self, audio_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            f.flush()
            audio_path = f.name
        waveform, sr = torchaudio.load(audio_path)
        os.unlink(audio_path)
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)
        waveform = waveform.squeeze(0)
        if waveform.dim() == 2:
            waveform = torch.mean(waveform, dim=0)
        chunk_lst = []
        for num_chuck in range(2):
            inputs = self.processor(waveform[num_chuck * 480000:(num_chuck+1) * 480000], sampling_rate=self.sampling_rate, return_tensors="pt")
            input_features = inputs.input_features.squeeze(0)
            chunk_lst.append(input_features)
        return chunk_lst

    @app.post("/eval", response_model=AudioAnalysisResponse)
    async def eval_endpoint(
        self,
        voice_file: UploadFile = File(...),
        agent_data: str = Form(...)  # for compatibility
    ):
        # Step 1: Validate and load audio file
        if not voice_file.filename.lower().endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only .wav files are supported")
        audio_bytes = await voice_file.read()
        try:
            chunk_lst = self.preprocess_audio(audio_bytes)
            input_features_1 = chunk_lst[0].unsqueeze(0).to(DEVICE)
            input_features_2 = chunk_lst[1].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.model(input_features_1, input_features_2)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > THRESHOLD).astype(bool).tolist()[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

        # Step 2: (Optional) Add transcription (placeholder for now)
        transcription = "<not_implemented>"

        return AudioAnalysisResponse(
            transcription=transcription,
            is_greeting=bool(preds[0]),
            is_introself=bool(preds[1]),
            is_informlicense=bool(preds[2]),
            is_informobjective=bool(preds[3]),
            is_informbenefit=bool(preds[4]),
            is_informinterval=bool(preds[5]),
        )

    @app.get("/")
    async def root(self):
        return {"msg": "WhisperClassifier API (Ray Serve) is running. POST audio to /eval"}

# ========== Ray Serve Start ==========
if __name__ == "__main__":
    ray.init(num_gpus=1)
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 4000, "timeout_keep_alive": 180})
    serve.run(WhisperClassifierAPI.bind(), route_prefix="/")
    print("Ray Serve is running on http://0.0.0.0:4000 (default)")
    import time
    while True:
        time.sleep(60)
