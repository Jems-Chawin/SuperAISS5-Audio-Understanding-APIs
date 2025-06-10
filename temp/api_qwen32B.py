'''
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
   Greeting Detection: 96.00%
   Self Introduction: 77.00%
   License Information: 100.00%
   Objective Information: 65.00%
   Benefit Information: 83.00%
   Interval Information: 97.00%

📈 F₂ Score Breakdown:
   Greeting F₂: 96.23%
   Self Introduction F₂: 21.98%
   License Information F₂: 100.00%
   Objective Information F₂: 87.72%
   Benefit Information F₂: 91.49%
   Interval Information F₂: 97.90%

📊 Detailed Metrics (where F₂ ≠ Accuracy):

   Greeting:
      Precision: 96.23%
      Recall: 96.23%
      True Positives: 51
      False Positives: 2
      False Negatives: 2

   Intro Self:
      Precision: 36.36%
      Recall: 20.00%
      True Positives: 4
      False Positives: 7
      False Negatives: 16

   Inform License:
      Precision: 100.00%
      Recall: 100.00%
      True Positives: 52
      False Positives: 0
      False Negatives: 0

   Inform Objective:
      Precision: 58.82%
      Recall: 100.00%
      True Positives: 50
      False Positives: 35
      False Negatives: 0

   Inform Benefit:
      Precision: 72.88%
      Recall: 97.73%
      True Positives: 43
      False Positives: 16
      False Negatives: 1

   Inform Interval:
      Precision: 96.55%
      Recall: 98.25%
      True Positives: 56
      False Positives: 2
      False Negatives: 1

🏆 Overall Performance:
   Overall Accuracy: 86.33%
   Average F₂ Score: 88.57%

💾 Detailed results saved to: load_test_results/multimodal_loadtest_results_20250529_174749.json
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
    "llm_model_name": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",  # Local OpenThaiGPT model
    
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
你是一名擅长评估银行客户经理（Relationship Manager，简称 RM）与客户之间对话的人工智能助手。

### 你的任务：
阅读 RM 与客户的完整对话内容，并根据以下六项标准，判断 RM 是否按照规范完成自我介绍。每项符合标准记为 1 分，未符合标准记为 0 分。

---

### 评估标准（共 6 项）：

1. **问候客户**
   - 必须以礼貌方式向客户打招呼。
   - 示例：”您好“、”早上好“、”哈喽，您好“。

2. **介绍自己的姓名（包括姓和名）**
   - 只需要评估 RM 是否完整介绍自己，不需要考虑客户的姓名。
   - 必须同时提供真实的名字和姓氏，昵称或仅名字不视为通过。
   - 如果名字与姓氏连在一起且没有空格（如“张三丰”），则必须明确标注为姓与名的组合，否则不通过。
   - 合格示例：“我叫张 伟”，“我是李 小美”。
   - 不合格示例：“我叫小张”，“你可以叫我小王”，“我叫阿明”。

3. **说明所持许可证类型及有效的许可证编号**
   - 必须明确提供许可证编号，并声明该许可证仍在有效期内。
   - 示例：“我持有编号为 12345 的金融执业许可证，目前仍在有效期内。”

4. **说明此次会面的目的**
   - 必须清楚说明此次会面目的，例如：投资组合更新、市场分析、新产品推荐等。
   - 不允许使用模糊或空泛的表述，如“想来聊聊”、“简单交流一下”。
   - 合格示例：“今天来是为了更新您的投资组合”、“我们来讨论一下当前市场走势”、“我想介绍一款新产品给您”。
   - 不合格示例：“今天是来按预约见面”、“我有些事想跟您聊聊”或完全未提及会面目的。

5. **说明客户从此次会面中可获得的好处**
   - 必须具体说明客户能从本次会面中获得什么收益。
   - 示例：“这将有助于您优化投资结构”、“可以帮助提高投资回报率”。

6. **说明此次会面预计所需时间**
   - 必须明确表示所需时间（可以是数字或文字），例如：15分钟、半小时、5分钟等。
   - 若仅说“很快”、“不久”等模糊表述，则不算通过。
   - 合格示例：“大约需要30分钟”、“预计聊1小时，您方便吗？”

---

### 评估方法：
1. 仔细阅读整段对话；
2. 针对每一条评估标准逐项判断；
3. 每项给出 1 或 0 分数，代表是否符合要求。

---

### 输出格式（必须遵循以下要求）：

请返回以下格式的 JSON，**字段名必须为泰语，值为 0 或 1，不得添加其他解释或文字说明**：

json
{
    "กล่าวสวัสดี": 1,
    "แนะนำชื่อและนามสกุล": 0,
    "บอกประเภทใบอนุญาตและเลขที่ใบอนุญาตที่ยังไม่หมดอายุ": 1,
    "บอกวัตถุประสงค์ของการเข้าพบครั้งนี้": 1,
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
        temp = CONFIG["llm_model_name"]
        logger.info(f"🚀 Loading {temp}...")
        
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
                    model_names=['MossFormer2_SE_48K']
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