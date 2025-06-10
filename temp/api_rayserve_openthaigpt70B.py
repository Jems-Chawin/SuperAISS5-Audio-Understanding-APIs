'''
data train.csv
================================================================================
🎯 MULTIMODAL LLM LOAD TEST RESULTS
================================================================================
📊 Request Performance:
   Total Requests: 100
   Successful: 93
   Failed: 7
   Timeouts: 0
   Success Rate: 93.00%
   Failure Rate: 7.00%
   Timeout Rate: 0.00%

🎯 Accuracy Metrics:
   Greeting Detection: 97.85%
   Self Introduction: 81.72%
   License Information: 100.00%
   Objective Information: 64.52%
   Benefit Information: 73.12%
   Interval Information: 98.92%

📈 F₂ Score Breakdown:
   Greeting F₂: 97.87%
   Self Introduction F₂: 72.92%
   License Information F₂: 100.00%
   Objective Information F₂: 86.96%
   Benefit Information F₂: 88.37%
   Interval Information F₂: 98.48%

📊 Detailed Metrics (where F₂ ≠ Accuracy):

   Greeting:
      Precision: 97.87%
      Recall: 97.87%
      True Positives: 46
      False Positives: 1
      False Negatives: 1

   Intro Self:
      Precision: 50.00%
      Recall: 82.35%
      True Positives: 14
      False Positives: 14
      False Negatives: 3

   Inform License:
      Precision: 100.00%
      Recall: 100.00%
      True Positives: 48
      False Positives: 0
      False Negatives: 0

   Inform Objective:
      Precision: 57.14%
      Recall: 100.00%
      True Positives: 44
      False Positives: 33
      False Negatives: 0

   Inform Benefit:
      Precision: 60.32%
      Recall: 100.00%
      True Positives: 38
      False Positives: 25
      False Negatives: 0

   Inform Interval:
      Precision: 100.00%
      Recall: 98.11%
      True Positives: 52
      False Positives: 0
      False Negatives: 1

🏆 Overall Performance:
   Overall Accuracy: 86.02%
   Average F₂ Score: 91.90%

💾 Detailed results saved to: load_test_results/multimodal_loadtest_results_20250529_215754.json
================================================================================
'''


"""
Production-ready serving with vLLM + Ray Serve for ASR + OpenThaiGPT pipeline
"""

import ray
from ray import serve
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from clearvoice import ClearVoice
import vllm
import re
import tempfile
import asyncio
from typing import Dict
import json
import os
import logging
from pydub import AudioSegment
from pathlib import Path
from pydantic import BaseModel
from fastapi import File, Form, UploadFile, HTTPException, Request
import concurrent.futures
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ray
ray.init(num_gpus=1)
serve.start(http_options={"host": "0.0.0.0", "port": 4000, "timeout_keep_alive": 180})

# Pydantic models (same as your original)
class AgentData(BaseModel):
    agent_fname: str
    agent_lname: str

class AudioAnalysisResponse(BaseModel):
    transcription: str
    is_greeting: bool
    is_introself: bool
    is_informlicense: bool
    is_informobjective: bool
    is_informbenefit: bool
    is_informinterval: bool

# Your system message
SYSTEM_MESSAGE = """
      คุณเป็น AI ที่เชี่ยวชาญในการประเมินการสนทนาของ Relationship Manager (RM) ธนาคารไทยพาณิชย์ (SCB) 
      งานของคุณคือวิเคราะห์บทสนทนาระหว่าง RM กับลูกค้า และประเมินว่า RM ได้ปฏิบัติตามมาตรฐานการแนะนำตัวครบถ้วนหรือไม่

      ## เกณฑ์การประเมิน (6 ข้อบังคับ):

      1. **กล่าวสวัสดี**
      - ต้องมีการทักทายลูกค้าอย่างสุภาพ
      - ต้องไม่ใช้คำทักทายที่ไม่สุภาพหรือไม่เหมาะสม
      - ตัวอย่าง: "สวัสดีครับ/ค่ะ", "สวัสดีตอนเช้าครับ", "หวัดดีครับ"
      - ไม่ผ่าน: "เว้ย", "ไง",

      2. **แนะนำชื่อและนามสกุล**
      - คิดแค่การแนะนำตัวเองของ RM เท่านั้น ไม่ต้องสนใจชื่อของลูกค้า
      - ต้องแนะนำชื่อจริงและนามสกุลครบถ้วน (ไม่ใช่แค่ชื่อเล่น)
      - หากมีแค่ชื่อ แต่ไม่มีนามสกุล จะถือว่าไม่ผ่าน
      - ชื่อกับนามสกุลอาจติดกันได้ ให้ดูตามบริบทของชื่ออย่างละเอียด เช่น "สมชายใจดี" มาจาก "สมชาย ใจดี" หรือ "นางสาวนิธินาทมันทนานนท์" มาจาก "นางสาวนิธินาท มันทนานนท์"
      - ตัวอย่าง: "ผมชื่อสมชาย นามสกุลใจดี", "ดิฉันชื่อสมหญิง สกุลดี"
      - ไม่ผ่าน: "ผมชื่อจอห์น", "เรียกผมว่าแมท", "ผมชื่อณัฐพล", "เรียกผมว่าธนพงษ์"

      3. **บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ**
      - ต้องระบุเลขที่ใบอนุญาต
      - ต้องยืนยันว่าใบอนุญาตยังไม่หมดอายุ
      - ต้องพูดทั้งเลขที่ใบอนุญาต + บอกว่ายังไม่หมดอายุ
      - หาก RM ไม่ได้พูดถึงใบอนุญาตเลย จะถือว่าผิด
      - หาก RM พูดถึงใบอนุญาต แต่ไม่ระบุเลขที่หรือหมดอายุ จะถือว่าผิด
      - ตัวอย่าง: "ผมมีใบอนุญาตเลขที่ 12345 ใบอนุญาตยังไม่หมดอายุครับ"
      - ไม่ผ่าน: "ผมมีใบอนุญาตเลขที่ 12345", "ใบอนุญาตของผมยังไม่หมดอายุ", "เลขที่ใบอนุญาตของผมคือ 12345"

      4. **บอกวัตถุประสงค์ของการเข้าพบครั้งนี้**
      - ต้องบอกเหตุผลที่มาพบลูกค้าให้ชัดเจนว่าเป็นเรื่องอะไร
      - ลูกค้าต้องรู้ว่าที่มาคุยคือคุยเรื่องอะไร
      - ต้องไม่ใช้คำพูดที่กว้างเกินไปหรือไม่ชัดเจน
      - ตัวอย่าง: "วันนี้มาเพื่ออัพเดทพอร์ตการลงทุน", "มาพูดคุยเรื่องสภาวะตลาด", "มาเสนอผลิตภัณฑ์ใหม่"
      - ไม่ผ่าน: "มาคุยเรื่องทั่วไป", "มาพบเพื่อพูดคุย", "มาคุยเรื่องการเงิน"

      5. **เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้**
      - ต้องอธิบายอย่างชัดเจนว่าลูกค้าจะได้อะไรจากการพบครั้งนี้
      - ต้องเน้นประโยชน์ที่ลูกค้าจะได้รับ ไม่ใช่แค่ RM ได้ประโยชน์
      - ต้องไม่ใช้คำพูดที่คลุมเครือหรือไม่ชัดเจน
      - ต้องไม่ใช้คำพูดที่เป็นการขายตรงหรือโฆษณาเกินจริง
      - ต้องไม่ใช้คำพูดที่เป็นการบังคับหรือกดดันลูกค้า
      - ตัวอย่าง: "ท่านจะได้ปรับสัดส่วนการลงทุนให้เหมาะสม", "จะช่วยเพิ่มผลตอบแทนจากการลงทุน"
      - ไม่ผ่าน: "", "อาจน่าสนใจ",

      6. **บอกระยะเวลาที่ใช้ในการเข้าพบ**
      - ต้องระบุเป็นช่วงเวลาเป็น **ตัวเลขเท่านั้น** (อาจเขียนในรูปตัวเลขหรือตัวหนังสือ) เช่น 15 นาที, ห้านาที ,ครึ่งชั่วโมง
      - หากบอกเพียงแค่ "แป๊บเดียว" หรือ "ไม่นาน" , "ไม่เกินชั่วโมง" จะถือว่าผิด
      - ต้องบอกระยะเวลาอย่างชัดเจน ไม่คลุมเครือ
      - ต้องบอกระยะเวลาเป็นช่วงเวลา ไม่ใช่แค่บอกว่า "จะใช้เวลานิดหน่อย"
      - ตัวอย่าง: "ขอเวลาประมาณ 30 นาที", "ใช้เวลาราว 1 ชั่วโมง สะดวกไหมครับ"
      - ไม่ผ่าน: "จะใช้เวลานิดหน่อย", "ไม่นาน", "แป๊บเดียว", "ไม่เกินชั่วโมง"

      ## วิธีการประเมิน:
      - อ่านบทสนทนาให้ละเอียด
      - ตรวจแต่ละข้อจากมุมมองของ RM เท่านั้น (อย่าดึงจากลูกค้า)
      - ให้ 1 หากพบข้อความตรงกับเกณฑ์
      - ให้ 0 หากไม่พบหรือคลุมเครือ

      ## รูปแบบ Output:
      ห้าม return คำอธิบายเพิ่มเติมเด็ดขาด
      ให้ return ผลลัพธ์เป็น JSON ตามรูปแบบนี้เท่านั้น:

      ```json
      {
          "กล่าวสวัสดี": 0,
          "แนะนำชื่อและนามสกุล": 0,
          "บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ": 0,
          "บอกวัตถุประสงค์ของการเข้าพบครั้งนี้": 0,
          "เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้": 0,
          "บอกระยะเวลาที่ใช้ในการเข้าพบ": 0
      }
"""

def analyze_transcription(text):
    license_patterns = {
        "investment": r"ใบอนุญาตแนะนำการลงทุน\s*(?:หมายเลข)?\s*([\w-]+)",
        "life_insurance": r"ใบอนุญาตแนะนำประกันชีวิต\s*(?:หมายเลข)?\s*([\w-]+)",
        "general_insurance": r"ใบอนุญาตแนะนำประกันวินาศภัย\s*(?:หมายเลข)?\s*([\w-]+)"
    }
    validity_pattern = r"(ยังไม่หมดอายุ|still active|not expired)"
    results = []
    for lic_type, pattern in license_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            license_number = matches[0]
            validity_match = re.search(validity_pattern, text, re.IGNORECASE)
            validity = "Valid" if validity_match or "หมดอายุ" not in text else "Expired"
            results.append({
                "license_type": lic_type,
                "license_number": license_number,
                "validity": validity
            })
    if not results:
        return [{"license_type": "None", "license_number": "N/A", "validity": "N/A"}]
    return results

def contains_time_mention(text):
    if text is None:
        return False
    text = str(text).lower()
    thai_numbers = {
        'ศูนย์': '0','หนึ่ง': '1', 'สอง': '2', 'สาม': '3', 'สี่': '4', 'ห้า': '5',
        'หก': '6', 'เจ็ด': '7', 'แปด': '8', 'เก้า': '9', 'สิบ': '10',
        'สิบเอ็ด': '11', 'สิบสอง': '12', 'สิบสาม': '13', 'สิบสี่': '14', 'สิบห้า': '15',
        'สิบหก': '16', 'สิบเจ็ด': '17', 'สิบแปด': '18', 'สิบเก้า': '19', 'ยี่สิบ': '20',
        'ยี่สิบเอ็ด': '21', 'ยี่สิบสอง': '22', 'ยี่สิบสาม': '23', 'ยี่สิบสี่': '24', 'ยี่สิบห้า': '25',
        'สามสิบ': '30', 'สี่สิบ': '40', 'ห้าสิบ': '50', 'หกสิบ': '60',
        'เจ็ดสิบ': '70', 'แปดสิบ': '80', 'เก้าสิบ': '90'
    }
    for thai_num, digit in thai_numbers.items():
        text = text.replace(thai_num, digit)
    true_patterns = [
        r'ขอเวลา\s*\d+\s*(?:นาที|ชั่วโมง)',
        r'ขอเวลา\s*\d+',
        r'ไม่เกิน\s*\d+\s*(?:นาที|ชั่วโมง)',
        r'ไม่เกิน\s*\d+',
        r'ไม่ถึง\s*\d+\s*(?:นาที|ชั่วโมง)',
        r'ไม่ถึง\s*\d+',
        r'เวลา\s*\d+\s*(?:นาที|ชั่วโมง)',
        r'ใช้เวลา\s*\d+\s*(?:นาที|ชั่วโมง)',
        r'\d+\s*นาที',
        r'\d+\s*ชั่วโมง',
        r'1\s*ชั่วโมง',
        r'ครึ่งชั่วโมง',
        r'ไม่ถึงครึ่งชั่วโมง',
        r'หนึ่งชั่วโมง',
        r'ขอเวลา.*?\d+.*?นาที',
        r'ขอเวลา.*?\d+.*?ชั่วโมง'
    ]
    for pattern in true_patterns:
        if re.search(pattern, text):
            return True
    false_patterns = [
        r'ขอเวลาไม่นาน',
        r'ขอเวลาแป๊ปเดียว',
        r'ขอเวลาสักแป๊ปนึง',
        r'ขอเวลาแป๊ปนึง',
        r'ขอเวลาสั้นๆ',
        r'ขอเวลาสั้นสั้น',
        r'ไม่เกินชั่วโมง',
        r'ไม่เกิน(?!\s*\d)',
        r'ขอเวลา(?!\s*\d)',
        r'ขอเวลาแป๊บ',
        r'ขอเวลาสักครู่',
        r'ขอเวลาแค่แป๊บเดียว'
    ]
    for pattern in false_patterns:
        if re.search(pattern, text):
            return False
    return False

def contains_benefit_keywords(text):
    if text is None:
        return False
    text = str(text).lower()
    benefit_keywords = ['ช่วยให้','ทำให้','เสริมสร้าง','ลดความเสี่ยง']
    return any(keyword in text for keyword in benefit_keywords)

# Ray Attributes
@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 5,
        "memory": 60 * 1024 * 1024 * 1024,  # 60GB per replica (or even less)
    },
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,  # Or 2 only if you have 2 H100s and >120GB RAM free
        "target_num_ongoing_requests_per_replica": 2,
    },
)

class ASRLLMService:
    def __init__(self):
        logger.info("Initializing ASR-LLM Service...")
        
        # Initialize ASR
        logger.info("🎤 Loading ASR model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_id = "nectec/Pathumma-whisper-th-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=444,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # Force Thai language
        self.asr_pipeline.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="th", task="transcribe"
        )
        
        logger.info("✅ ASR model loaded successfully")
        
        # Initialize vLLM for OpenThaiGPT
        logger.info("🚀 Loading OpenThaiGPT 1.5 72B with vLLM...")
        
        self.llm = LLM(
            model="/home/siamai/data/models/openthaigpt1.5-72b-instruct",
            tensor_parallel_size=1,  # Use 1 GPU, increase if you have multiple
            gpu_memory_utilization=0.70,  # Use 85% of GPU memory
            max_model_len=3000,  # Reduced from 5000 to fit in memory
            dtype="bfloat16",
            quantization="bitsandbytes",  # 4-bit quantization
            load_format="auto",
            trust_remote_code=True,
            download_dir="/tmp/model_cache",
            enforce_eager=False,  # Use CUDA graphs for better performance
        )
        
        # Load tokenizer for chat template
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/siamai/data/models/openthaigpt1.5-72b-instruct",
            trust_remote_code=True,
        )
        
        logger.info("✅ vLLM model loaded successfully")
        
        # Initialize ClearVoice
        logger.info("🔊 Loading ClearVoice model...")
        try:
            self.clearvoice = ClearVoice(
                task='speech_separation',
                model_names=['MossFormer2_SE_48K']
            )
            self.enable_speech_enhancement = True
            logger.info("✅ ClearVoice loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ClearVoice: {e}")
            self.enable_speech_enhancement = False
        
        self.system_message = SYSTEM_MESSAGE
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "failed_requests": 0,
            "total_time": 0.0,
        }
    
    async def __call__(self, request: List[Request]):
        """
        Main endpoint that processes audio and returns analysis
        """
        form = await request.form()
        voice_file = form["voice_file"]   # UploadFile
        agent_data = form["agent_data"]   # str
    
        import time
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        temp_file_path = None
        
        try:
            # Validate input
            if not voice_file.filename or not any(
                voice_file.filename.lower().endswith(ext)
                for ext in ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm']
            ):
                raise HTTPException(status_code=400, detail="Invalid audio file format")
            
            # Parse agent data
            try:
                agent_info = AgentData(**json.loads(agent_data))
            except:
                raise HTTPException(status_code=400, detail="Invalid agent data format")
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(voice_file.filename).suffix) as tmp:
                temp_file_path = tmp.name
                content = await voice_file.read()
                tmp.write(content)
            
            # Process audio
            transcription = await self._transcribe_audio(temp_file_path)
            # Analyze with LLM
            analysis = await self._analyze_with_llm(transcription)

            # ---- POSTPROCESS LOGIC ----
            license_info = analyze_transcription(transcription)
            has_time = contains_time_mention(transcription)
            has_benefit = contains_benefit_keywords(transcription)

            # OVERRIDE FIELDS ACCORDING TO YOUR RULES:
            # Example: If postprocess finds a valid license, set is_informlicense = True
            if any(
                lic["license_type"] != "None" and lic["validity"] == "Valid"
                for lic in license_info
            ):
                analysis["is_informlicense"] = True

            # If postprocess detects specific time, set is_informinterval = True
            if has_time:
                analysis["is_informinterval"] = True

            # If postprocess detects benefit keywords, set is_informbenefit = True
            if has_benefit:
                analysis["is_informbenefit"] = True
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["total_time"] += processing_time
            
            # Log performance every 10 requests
            if self.metrics["total_requests"] % 10 == 0:
                avg_time = self.metrics["total_time"] / self.metrics["total_requests"]
                logger.info(f"Avg processing time: {avg_time:.2f}s")
            
            # Log or record what was changed, if you like
            logger.info(f"Corrected analysis by postprocess: {analysis}")
            
            return AudioAnalysisResponse(
                transcription=transcription,
                **analysis
            )
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            logger.error(f"Processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def _transcribe_audio(self, audio_file_path: str) -> str:
        audio = AudioSegment.from_file(audio_file_path)
        audio_length_ms = len(audio)
        if audio_length_ms <= 30000:
            result = self.asr_pipeline(audio_file_path)
            return result["text"].strip()
        
        chunk_length_ms = 27000
        stride_length_ms = 2000
        chunks = []
        for start_ms in range(0, audio_length_ms, chunk_length_ms - stride_length_ms):
            end_ms = min(start_ms + chunk_length_ms, audio_length_ms)
            chunk = audio[start_ms:end_ms]
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            chunk.export(tmp_file.name, format="wav", parameters=["-ar", "16000"])
            chunks.append(tmp_file.name)
            tmp_file.close()
        
        def transcribe_chunk(chunk_path):
            try:
                result = self.asr_pipeline(chunk_path)
                return result["text"].strip()
            finally:
                os.unlink(chunk_path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            texts = list(executor.map(transcribe_chunk, chunks))
        return " ".join(texts)
    
    async def _analyze_with_llm(self, transcription: str) -> Dict[str, bool]:
        """Analyze transcription using vLLM"""
        try:
            # Skip very short transcriptions
            if len(transcription) < 10 or transcription == "transcription_error":
                return self._get_default_analysis()
            
            # Truncate long transcriptions
            if len(transcription) > 4000:
                transcription = transcription[:4000] + "..."
            
            # Prepare prompt
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"บทสนทนา: {transcription}"}
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            
            # vLLM sampling parameters
            sampling_params = SamplingParams(
                temperature=1.0,  # Deterministic
                max_tokens=200,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
            )
            
            # Generate with vLLM
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)
            response_text = outputs[0].outputs[0].text
            # print(f"===> Debug responsetxt: {response_text}")
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                
                return {
                    "is_greeting": bool(result.get('กล่าวสวัสดี', 0)),
                    "is_introself": bool(result.get('แนะนำชื่อและนามสกุล', 0)),
                    "is_informlicense": bool(result.get('บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ', 0)),
                    "is_informobjective": bool(result.get('บอกวัตถุประสงค์ของการเข้าพบครั้งนี้', 0)),
                    "is_informbenefit": bool(result.get('เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้', 0)),
                    "is_informinterval": bool(result.get('บอกระยะเวลาที่ใช้ในการเข้าพบ', 0)),
                }
            else:
                logger.error(f"Failed to parse JSON from response: {response_text}")
                return self._get_default_analysis()
                
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, bool]:
        """Return default analysis (all False)"""
        return {
            "is_greeting": False,
            "is_introself": False,
            "is_informlicense": False,
            "is_informobjective": False,
            "is_informbenefit": False,
            "is_informinterval": False,
        }

# Additional endpoints for monitoring
@serve.deployment
class HealthCheck:
    async def __call__(self):
        return {
            "status": "healthy",
            "ray_cluster": ray.cluster_resources(),
            "timestamp": str(asyncio.get_event_loop().time())
        }

# Deploy the services
deployment = ASRLLMService.bind()
health_deployment = HealthCheck.bind()

# Run both deployments with route prefixes
serve.run(deployment, name="asr_llm_service", route_prefix="/eval")
serve.run(health_deployment, name="health_check", route_prefix="/health")

if __name__ == "__main__":
    # Keep the script running
    import time
    logger.info("ASR-LLM Service with vLLM is running...")
    logger.info("Endpoints:")
    logger.info("  - POST http://0.0.0.0:4000/eval (main service)")
    logger.info("  - GET  http://0.0.0.0:4000/health (health check)")
    
    try:
        while True:
            time.sleep(60)
            # Optional: Add periodic health checks or metrics logging
    except KeyboardInterrupt:
        logger.info("Shutting down service...")
        serve.shutdown()