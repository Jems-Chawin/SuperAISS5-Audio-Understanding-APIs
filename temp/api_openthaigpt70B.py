'''
================================================================================
🎯 MULTIMODAL LLM LOAD TEST RESULTS
================================================================================
📊 Request Performance:
   Total Requests: 100
   Successful: 99
   Failed: 1
   Timeouts: 0
   Success Rate: 99.00%
   Failure Rate: 1.00%
   Timeout Rate: 0.00%

🎯 Accuracy Metrics:
   Greeting Detection: 96.97%
   Self Introduction: 86.87%
   License Information: 98.99%
   Objective Information: 65.66%
   Benefit Information: 74.75%
   Interval Information: 100.00%

📈 F₂ Score Breakdown:
   Greeting F₂: 96.59%
   Self Introduction F₂: 84.91%
   License Information F₂: 98.46%
   Objective Information F₂: 87.81%
   Benefit Information F₂: 89.58%
   Interval Information F₂: 100.00%

📊 Detailed Metrics (where F₂ ≠ Accuracy):

   Greeting:
      Precision: 98.08%
      Recall: 96.23%
      True Positives: 51
      False Positives: 1
      False Negatives: 2

   Intro Self:
      Precision: 60.00%
      Recall: 94.74%
      True Positives: 18
      False Positives: 12
      False Negatives: 1

   Inform License:
      Precision: 100.00%
      Recall: 98.08%
      True Positives: 51
      False Positives: 0
      False Negatives: 1

   Inform Objective:
      Precision: 59.04%
      Recall: 100.00%
      True Positives: 49
      False Positives: 34
      False Negatives: 0

   Inform Benefit:
      Precision: 63.24%
      Recall: 100.00%
      True Positives: 43
      False Positives: 25
      False Negatives: 0

   Inform Interval:
      Precision: 100.00%
      Recall: 100.00%
      True Positives: 56
      False Positives: 0
      False Negatives: 0

🏆 Overall Performance:
   Overall Accuracy: 87.21%
   Average F₂ Score: 92.57%

💾 Detailed results saved to: load_test_results/multimodal_loadtest_results_20250529_142645.json
================================================================================
'''


"""
High-performance ASR FastAPI server with Unsloth-optimized LLM for text analysis.
Optimized for load testing with Locust - handles 10+ concurrent requests for 10 minutes.
"""

import json
import os
import tempfile
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import uuid
import gc

import torch
from pydub import AudioSegment
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from clearvoice import ClearVoice
import numpy as np
from unsloth import FastLanguageModel
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    # Model settings
    "asr_model_name": "nectec/Pathumma-whisper-th-large-v3",
    "llm_model_name": "/home/siamai/data/models/openthaigpt1.5-72b-instruct",  # Local OpenThaiGPT model
    
    # Performance settings
    "chunk_length_ms": 30000,  # 30 second chunks - optimal for Thai
    "stride_length_ms": 5000,  # 5 second overlap to preserve context
    "max_file_size_mb": 50,
    "max_workers": 8,
    "enable_speech_enhancement": True,
    
    # Timeouts
    "request_timeout": 180,  # 180 seconds as required
    "asr_timeout": 180,  # Increased ASR timeout
    "llm_timeout": 180,
    "enhancement_timeout": 180,
    
    # Unsloth settings
    "max_seq_length": 5000,  # Increased for larger context
    "dtype": torch.bfloat16,  # Better for large models
    "load_in_4bit": True,
    "max_new_tokens": 200,
    "use_cache": True,
}

# Pydantic models (keeping exact same structure)
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

# Global variables
asr_pipeline = None
llm_model = None
llm_tokenizer = None

clearvoice_model = None
executor = None
model_lock = asyncio.Lock()

# Performance metrics
metrics = {
    "total_requests": 0,
    "failed_requests": 0,
    "total_time": 0.0,
    "active_requests": 0,
}

# Optimized system prompt (same as before)
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

def initialize_models():
    """Initialize all models with optimizations."""
    global asr_pipeline, llm_model, llm_tokenizer, vllm_model, clearvoice_model, executor
    
    try:
        # Initialize thread pool
        executor = ThreadPoolExecutor(max_workers=CONFIG["max_workers"])
        
        # Initialize ASR model
        logger.info("🎤 Loading ASR model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load ASR model with optimizations
        model_id = CONFIG["asr_model_name"]
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=444,  # Safe limit for Whisper (448 - 4 for special tokens)
            chunk_length_s=30,
            batch_size=16,  # Increased batch size
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # Force Thai language
        asr_pipeline.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="th", task="transcribe"
        )
        
        logger.info("✅ ASR model loaded successfully")
        
        # Initialize OpenThaiGPT with Unsloth or fallback to transformers
        logger.info("🚀 Loading OpenThaiGPT 1.5 72B...")
        
        # Check if we should use direct transformers loading for large models
        USE_DIRECT_TRANSFORMERS = os.getenv("USE_DIRECT_TRANSFORMERS", "false").lower() == "true"
        
        if USE_DIRECT_TRANSFORMERS:
            # Direct transformers loading with CPU offloading for 72B model
            logger.info("Using direct transformers loading with CPU offloading...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Configure quantization with CPU offloading
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            # Load tokenizer
            llm_tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model_name"])
            
            # Load model with explicit device map
            llm_model = AutoModelForCausalLM.from_pretrained(
                CONFIG["llm_model_name"],
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "65GiB", "cpu": "100GiB"},  # Leave room for activations
                offload_folder="offload",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            
            logger.info("✅ Model loaded with transformers and CPU offloading")
            
        else:
            # Try Unsloth loading
            try:
                # For 72B, we need to be very conservative with memory
                logger.info("Attempting to load with Unsloth (this may fail for 72B models)...")
                
                # Try with minimal settings
                llm_model, llm_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=CONFIG["llm_model_name"],
                    max_seq_length=5000,  # Very short to save memory
                    dtype=None,  # Let Unsloth decide
                    load_in_4bit=True,
                )
                
                llm_model = FastLanguageModel.for_inference(llm_model)
                logger.info("✅ OpenThaiGPT loaded successfully with Unsloth")
                
            except Exception as e:
                logger.error(f"Unsloth loading failed: {e}")
                logger.error("\n" + "="*60)
                logger.error("72B model is too large for Unsloth on single GPU")
                logger.error("Please use one of these options:")
                logger.error("1. Set USE_DIRECT_TRANSFORMERS=true for CPU offloading")
                logger.error("2. Use a smaller model (14B or 32B)")
                logger.error("3. Use vLLM for better memory management")
                logger.error("="*60 + "\n")
                raise
        
        # Initialize ClearVoice if enabled
        if CONFIG["enable_speech_enhancement"]:
            logger.info("🔊 Loading ClearVoice model...")
            try:
                clearvoice_model = ClearVoice(
                    task='speech_separation',
                    model_names=['MossFormer2_SS_16K']
                )
                logger.info("✅ ClearVoice loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ClearVoice: {e}")
                CONFIG["enable_speech_enhancement"] = False
        
        # Warm up models
        logger.info("🔥 Warming up models...")
        asyncio.create_task(warmup_models())
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

async def warmup_models():
    """Warm up models with dummy requests."""
    try:
        # Warm up LLM
        dummy_text = "สวัสดีครับ ผมชื่อทดสอบ"
        await analyze_with_llm(dummy_text)
        logger.info("✅ Models warmed up")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")

async def transcribe_audio_async(audio_file_path: str) -> str:
    """Async wrapper for audio transcription."""
    loop = asyncio.get_event_loop()
    
    def transcribe():
        try:
            # For direct file processing without chunking (faster)
            result = asr_pipeline(
                audio_file_path,
                chunk_length_s=30,  # Process in 30s chunks
                batch_size=16,
                return_timestamps=False,
                generate_kwargs={
                    "language": "th",
                    "task": "transcribe",
                    "num_beams": 1,  # Faster greedy decoding
                    "max_new_tokens": 444,  # Safe limit for Whisper
                }
            )
            return result["text"].strip()
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            # Fallback to manual chunking if direct processing fails
            try:
                audio = AudioSegment.from_file(audio_file_path)
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio_length_ms = len(audio)
                
                if audio_length_ms <= CONFIG["chunk_length_ms"]:
                    result = asr_pipeline(audio_file_path)
                    return result["text"].strip()
                
                # Process in chunks with overlap
                transcriptions = []
                chunk_length_ms = CONFIG["chunk_length_ms"]
                stride_length_ms = CONFIG.get("stride_length_ms", 5000)
                
                # Calculate overlap to preserve context
                for start_ms in range(0, audio_length_ms, chunk_length_ms - stride_length_ms):
                    end_ms = min(start_ms + chunk_length_ms, audio_length_ms)
                    chunk = audio[start_ms:end_ms]
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        chunk_path = tmp_file.name
                        chunk.export(chunk_path, format="wav", parameters=["-ar", "16000"])
                    
                    try:
                        result = asr_pipeline(
                            chunk_path,
                            generate_kwargs={
                                "language": "th",
                                "task": "transcribe",
                                "max_new_tokens": 444,
                            }
                        )
                        
                        # For overlapping chunks, handle deduplication
                        text = result["text"].strip()
                        if transcriptions and stride_length_ms > 0:
                            # Simple deduplication - you might want more sophisticated approach
                            # This removes potential duplicate words at boundaries
                            last_words = transcriptions[-1].split()[-5:] if transcriptions else []
                            new_words = text.split()
                            
                            # Find overlap and merge
                            overlap_start = 0
                            for i in range(min(5, len(new_words))):
                                if new_words[:i+1] == last_words[-i-1:]:
                                    overlap_start = i + 1
                            
                            if overlap_start > 0:
                                text = " ".join(new_words[overlap_start:])
                        
                        if text:
                            transcriptions.append(text)
                    finally:
                        try:
                            os.unlink(chunk_path)
                        except:
                            pass
                
                return " ".join(transcriptions)
            except Exception as e2:
                logger.error(f"Fallback transcription also failed: {e2}")
                return "transcription_error"
    
    # Run in thread pool with timeout
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, transcribe),
            timeout=CONFIG["asr_timeout"]
        )
    except asyncio.TimeoutError:
        logger.error("ASR timeout")
        return "asr_timeout_error"

def run_unsloth_inference(transcription: str) -> str:
    """Run inference using Unsloth optimized model or direct transformers."""
    try:
        # Format messages for chat
        temp = transcription.replace("คุณ", "")
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": f"บทสนทนา: {temp}"}
        ]
        
        # Apply chat template
        inputs = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        
        # Handle both Unsloth and direct transformers models
        if hasattr(llm_model, 'device'):
            # Unsloth model
            inputs = inputs.to(llm_model.device)
        else:
            # Direct transformers model - inputs go to first device
            inputs = inputs.to("cuda:0")

        # Generate with optimized settings
        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids=inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=1.0,  # Deterministic
                do_sample=False,
                use_cache=CONFIG["use_cache"],
                pad_token_id=llm_tokenizer.eos_token_id,
            )

        # Decode response
        response = llm_tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise

async def analyze_with_llm(transcription: str) -> Dict[str, bool]:
    """Analyze transcription using OpenThaiGPT with Unsloth."""
    try:
        print(f"===>Transcription:\n{transcription}")
        # Skip very short transcriptions
        if len(transcription) < 10 or transcription in ["transcription_error", "asr_timeout_error"]:
            return _get_default_analysis()
        
        # Truncate long transcriptions
        if len(transcription) > 4000:  # Adjusted for larger context
            transcription = transcription[:4000] + "..."
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        async with model_lock:  # Ensure thread-safe model access
            response = await loop.run_in_executor(
                executor, 
                run_unsloth_inference, 
                transcription
            )
        
        # Extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            ret = {
                "is_greeting": bool(result.get('กล่าวสวัสดี', 0)),
                "is_introself": bool(result.get('แนะนำชื่อและนามสกุล', 0)),
                "is_informlicense": bool(result.get('บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ', 0)),
                "is_informobjective": bool(result.get('บอกวัตถุประสงค์ของการเข้าพบครั้งนี้', 0)),
                "is_informbenefit": bool(result.get('เน้นประโยชน์ว่าลูกค้าได้ประโยชน์อะไรจากการเข้าพบครั้งนี้', 0)),
                "is_informinterval": bool(result.get('บอกระยะเวลาที่ใช้ในการเข้าพบ', 0)),
            }
            print(f"===>Return:\n{ret}")
            
            return ret
        else:
            logger.error(f"Failed to parse JSON from response: {response}")
            return _get_default_analysis()
            
    except Exception as e:
        logger.error(f"LLM analysis error: {e}")
        return _get_default_analysis()

def _get_default_analysis() -> Dict[str, bool]:
    """Return default analysis (all False)."""
    return {
        "is_greeting": False,
        "is_introself": False,
        "is_informlicense": False,
        "is_informobjective": False,
        "is_informbenefit": False,
        "is_informinterval": False,
    }

# Initialize FastAPI app
app = FastAPI(
    title="High-Performance ASR+LLM API with OpenThaiGPT",
    version="3.0.0",
    docs_url=None,  # Disable docs for performance
    redoc_url=None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    initialize_models()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if executor:
        executor.shutdown(wait=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Post method - evaluation (blind test)
@app.post("/eval", response_model=AudioAnalysisResponse)
async def evaluate_audio(
    background_tasks: BackgroundTasks,
    voice_file: UploadFile = File(...),
    agent_data: str = Form(...)
):
    """
    Evaluate audio file for RM compliance.
    Returns transcription and 6 boolean criteria.
    """
    start_time = time.time()
    temp_file_path = None
    
    # Update metrics
    metrics["total_requests"] += 1
    metrics["active_requests"] += 1
    
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
        
        # Check models
        if not asr_pipeline or not llm_model:
            raise HTTPException(status_code=503, detail="Models not ready")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(voice_file.filename).suffix) as tmp:
            temp_file_path = tmp.name
            content = await voice_file.read()
            tmp.write(content)
        
        # Process audio with timeout
        async def process_audio():
            # Transcribe audio
            transcription = await transcribe_audio_async(temp_file_path)
            
            # Analyze with LLM
            analysis = await analyze_with_llm(transcription)
            
            return AudioAnalysisResponse(
                transcription=transcription,
                **analysis
            )
        
        # Apply request timeout
        result = await asyncio.wait_for(
            process_audio(),
            timeout=CONFIG["request_timeout"]
        )
        
        # Update metrics
        processing_time = time.time() - start_time
        metrics["total_time"] += processing_time
        
        # Log performance
        if metrics["total_requests"] % 10 == 0:
            avg_time = metrics["total_time"] / metrics["total_requests"]
            logger.info(f"Avg processing time: {avg_time:.2f}s")
        
        return result
        
    except asyncio.TimeoutError:
        metrics["failed_requests"] += 1
        logger.error(f"Request timeout after {CONFIG['request_timeout']}s")
        return AudioAnalysisResponse(
            transcription="timeout_error",
            **_get_default_analysis()
        )
    except HTTPException:
        metrics["failed_requests"] += 1
        raise
    except Exception as e:
        metrics["failed_requests"] += 1
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        metrics["active_requests"] -= 1
        if temp_file_path:
            background_tasks.add_task(lambda: os.unlink(temp_file_path) if os.path.exists(temp_file_path) else None)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models": {
            "asr": asr_pipeline is not None,
            "llm": llm_model is not None,
            "clearvoice": clearvoice_model is not None,
        },
        "metrics": {
            "total_requests": metrics["total_requests"],
            "failed_requests": metrics["failed_requests"],
            "active_requests": metrics["active_requests"],
            "avg_processing_time": metrics["total_time"] / max(1, metrics["total_requests"]),
            "success_rate": 1 - (metrics["failed_requests"] / max(1, metrics["total_requests"])),
        }
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ASR+OpenThaiGPT API is running. POST to /eval to analyze audio."}

if __name__ == "__main__":
    # Run with optimal settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4000,
        workers=1,  # Single worker for model sharing
        loop="uvloop" if os.name != 'nt' else "asyncio",
        access_log=False,
        log_level="info",
        limit_concurrency=100,  # Handle many concurrent connections
    )