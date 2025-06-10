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
   Greeting Detection: 97.00%
   Self Introduction: 75.00%
   License Information: 100.00%
   Objective Information: 56.00%
   Benefit Information: 79.00%
   Interval Information: 84.00%

📈 F₂ Score Breakdown:
   Greeting F₂: 96.59%
   Self Introduction F₂: 80.00%
   License Information F₂: 100.00%
   Objective Information F₂: 85.03%
   Benefit Information F₂: 89.96%
   Interval Information F₂: 93.65%

📊 Detailed Metrics (where F₂ ≠ Accuracy):

   Greeting:
      Precision: 98.08%
      Recall: 96.23%
      True Positives: 51
      False Positives: 1
      False Negatives: 2

   Intro Self:
      Precision: 44.44%
      Recall: 100.00%
      True Positives: 20
      False Positives: 25
      False Negatives: 0

   Inform License:
      Precision: 100.00%
      Recall: 100.00%
      True Positives: 52
      False Positives: 0
      False Negatives: 0

   Inform Objective:
      Precision: 53.19%
      Recall: 100.00%
      True Positives: 50
      False Positives: 44
      False Negatives: 0

   Inform Benefit:
      Precision: 68.25%
      Recall: 97.73%
      True Positives: 43
      False Positives: 20
      False Negatives: 1

   Inform Interval:
      Precision: 78.87%
      Recall: 98.25%
      True Positives: 56
      False Positives: 15
      False Negatives: 1

🏆 Overall Performance:
   Overall Accuracy: 81.83%
   Average F₂ Score: 89.70%

💾 Detailed results saved to: load_test_results/multimodal_loadtest_results_20250528_065257.json
================================================================================
'''


"""
Simple script to run the ASR FastAPI server.
This avoids module import issues by running everything in the same process.
"""

import json
import os
import re
import tempfile
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from io import BytesIO
from contextlib import asynccontextmanager

import torch
import pandas as pd
from difflib import SequenceMatcher
from pydub import AudioSegment
from transformers import pipeline
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 4000,
    "enable_gpu": True,
    "chunk_length_ms": 27000,  # 27 seconds per chunk
    "model_name": "nectec/Pathumma-whisper-th-large-v3",
    "language": "th",
    "task": "transcribe",
    "max_file_size_mb": 100,
    "enable_detailed_logging": True,
    "cleanup_temp_files": True,
}

# Pydantic models
class AgentData(BaseModel):
    """Agent information structure."""
    agent_fname: str
    agent_lname: str

class AudioAnalysisResponse(BaseModel):
    """Response structure for audio analysis."""
    is_greeting: bool
    is_introself: bool
    is_informlicense: bool
    is_informobjective: bool
    is_informbenefit: bool
    is_informinterval: bool
    
    # Additional metadata
    agent_name: str
    transcription: str
    audio_duration_seconds: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    confidence_score: Optional[float] = None
    timestamp: str

class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    model_loaded: bool
    gpu_available: bool

# Global variables
asr_pipeline = None
server_start_time = datetime.now()

class ASRProcessor:
    """Handles automatic speech recognition processing."""
    
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.torch_dtype = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ASR model pipeline."""
        try:
            logger.info("🤖 Initializing ASR model...")
            
            # Configure device and dtype
            self.device = 0 if torch.cuda.is_available() and SERVER_CONFIG["enable_gpu"] else -1
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
            logger.info(f"Using dtype: {self.torch_dtype}")
            
            # Load model pipeline
            self.pipeline = pipeline(
                task="automatic-speech-recognition",
                model=SERVER_CONFIG["model_name"],
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            # Configure language and task
            self.pipeline.model.config.forced_decoder_ids = self.pipeline.tokenizer.get_decoder_prompt_ids(
                language=SERVER_CONFIG["language"], 
                task=SERVER_CONFIG["task"]
            )
            
            logger.info("✅ ASR model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ASR model: {e}")
            raise RuntimeError(f"ASR model initialization failed: {e}")
    
    async def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Transcribe audio file to text using chunking approach.
        """
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            full_transcription = ""
            
            # Calculate number of chunks
            chunk_length_ms = SERVER_CONFIG["chunk_length_ms"]
            num_chunks = (len(audio) + chunk_length_ms - 1) // chunk_length_ms
            
            logger.info(f"Processing {num_chunks} chunks for audio file")
            
            # Process each chunk
            for i in range(num_chunks):
                start = i * chunk_length_ms
                chunk = audio[start:start + chunk_length_ms]
                
                # Create temporary chunk file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    chunk_path = temp_file.name
                    chunk.export(chunk_path, format="wav")
                
                try:
                    # Run transcription in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None, 
                        self.pipeline, 
                        chunk_path
                    )
                    
                    chunk_text = output["text"].strip()
                    full_transcription += chunk_text + " "
                    
                    logger.debug(f"Chunk {i+1}/{num_chunks}: {len(chunk_text)} characters")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    full_transcription += "[ERROR] "
                
                finally:
                    # Cleanup temporary file
                    if SERVER_CONFIG["cleanup_temp_files"]:
                        try:
                            os.unlink(chunk_path)
                        except:
                            pass
            
            return full_transcription.strip()
            
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            raise

class TextAnalyzer:
    """Analyzes transcribed text for specific patterns and content."""
    
    @staticmethod
    def analyze_speech_content(transcription: str, agent_data: AgentData) -> Dict[str, bool]:
        """
        Analyze transcription for specific speech patterns.
        """
        text = str(transcription).lower()
        agent_first = str(agent_data.agent_fname).lower()
        agent_last = str(agent_data.agent_lname).lower()
        
        analysis_result = {
            "is_greeting": TextAnalyzer._detect_greeting(text),
            "is_introself": TextAnalyzer._detect_self_introduction(text, agent_first, agent_last),
            "is_informlicense": TextAnalyzer._detect_license_info(text),
            "is_informobjective": TextAnalyzer._detect_objective_info(text),
            "is_informbenefit": TextAnalyzer._detect_benefit_info(text),
            "is_informinterval": TextAnalyzer._detect_interval_info(text),
        }
        
        return analysis_result
    
    @staticmethod
    def _detect_greeting(text: str) -> bool:
        """Detect greeting patterns in text."""
        greeting_patterns = [
            r'สวัสดี',
            r'หวัดดี',
            r'ฮัลโหล',
        ]
        
        for pattern in greeting_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def _normalize_thai_text(text: str) -> str:
        """
        Normalize Thai text by removing tone marks and silent letters that ASR often misses.
        """
        # Remove common Thai tone marks and silent letters that ASR struggles with
        text = re.sub(r'[์็่้๊๋]', '', text)  # Remove tone marks and silent marker
        
        # Common ASR substitutions for Thai
        substitutions = {
            'ฒ': 'ต',     # ฒ often transcribed as ต
            'ฎ': 'ด',     # ฎ often transcribed as ด  
            'ฏ': 'ต',     # ฏ often transcribed as ต
            'ญ': 'ย',     # ญ often transcribed as ย
            'ฌ': 'ช',     # ฌ often transcribed as ช
            'ธ': 'ท',     # ธ often transcribed as ท
            'ภ': 'พ',     # ภ often transcribed as พ
            'ฟ': 'พ',     # ฟ often transcribed as พ
        }
        
        for old, new in substitutions.items():
            text = text.replace(old, new)
        
        return text.lower()

    @staticmethod  
    def _fuzzy_name_match(asr_text: str, true_name: str, threshold: float = 0.8) -> bool:
        """
        Check if ASR text contains a fuzzy match for the true name.
        Uses normalized text and similarity scoring.
        """
        # Normalize both texts
        normalized_asr = TextAnalyzer._normalize_thai_text(asr_text)
        normalized_name = TextAnalyzer._normalize_thai_text(true_name)
        
        # Method 1: Direct substring match after normalization
        if normalized_name in normalized_asr:
            return True
        
        # Method 2: Sliding window similarity check
        name_len = len(normalized_name)
        if name_len < 2:  # Too short to meaningfully match
            return False
        
        # Check all possible substrings of similar length
        for i in range(len(normalized_asr) - name_len + 1):
            window = normalized_asr[i:i + name_len]
            similarity = SequenceMatcher(None, normalized_name, window).ratio()
            
            if similarity >= threshold:
                return True
        
        # Method 3: Check with slight length variations (±1 character)
        for length_offset in [-1, 0, 1]:
            target_len = max(1, name_len + length_offset)
            for i in range(len(normalized_asr) - target_len + 1):
                window = normalized_asr[i:i + target_len]
                similarity = SequenceMatcher(None, normalized_name, window).ratio()
                
                if similarity >= threshold:
                    return True
        
        return False
    
    @staticmethod
    def _detect_self_introduction(text: str, agent_first: str, agent_last: str) -> bool:
        """
        Fuzzy matching version for ASR text with potential name transcription errors.
        """
        text_lower = text.lower()
        
        # Use fuzzy matching for names
        has_first_name = TextAnalyzer._fuzzy_name_match(text_lower, agent_first, threshold=0.5)
        has_last_name = TextAnalyzer._fuzzy_name_match(text_lower, agent_last, threshold=0.5)
        
        # MUST have both names (fuzzy matched), otherwise False
        if not (has_first_name and has_last_name):
            return False
        
        # Since both names are present (fuzzy matched), check for introduction patterns
        introduction_patterns = [
            # Strong patterns
            r'(ผม|ดิฉัน).*ชื่อ',
            r'ชื่อ.*(ผม|ดิฉัน)',
            r'สวัสดี.*(ผม|ดิฉัน)',
            r'(ผม|ดิฉัน).*สวัสดี',
            r'แนะนำ.*ตัว',
            r'แนะนำ.*(ผม|ดิฉัน)',
            r'รู้จัก.*(ผม|ดิฉัน)',
            r'(ผม|ดิฉัน).*รู้จัก',
            
            # Basic introduction indicators
            r'(ผม|ดิฉัน)', r'ชื่อ', r'สวัสดี', r'แนะนำ', 
            r'รู้จัก', r'คือ', r'เป็น', r'บอก', r'เรียก',
            
            # Polite particles often used in introductions
            r'ครับ', r'ค่ะ', r'คะ',
        ]
        
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in introduction_patterns)
    
    @staticmethod
    def _detect_license_info(text: str) -> bool:
        """Detect license/credential information."""
        license_patterns = [
            r'ใบอนุญาต',
            r'เลขที่',
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in license_patterns)
    
    @staticmethod
    def _detect_objective_info(text: str) -> bool:
        """
        F₂-optimized: Try to get back to 95%+ recall while maintaining 55%+ precision
        Current: 90% recall, 5 false negatives
        """
        text_lower = text.lower()
        
        # TIER 1: High confidence patterns (maintain precision)
        high_confidence = [
            r'วัตถุประสงค์',
            r'จุดประสงค์', 
            r'เข้าพบ.*เพื่อ.*(?:อธิบาย|แนะนำ|บอก)',
            r'มาพบ.*เพื่อ.*(?:อธิบาย|แนะนำ|บอก)',
            r'(?:อธิบาย|แนะนำ).*(?:การลงทุน|ลงทุน|พอร์ต|กองทุน|ผลิตภัณฑ์)',
            r'(?:วันนี้|ครั้งนี้).*(?:มา|เข้า).*เพื่อ.*(?:อธิบาย|แนะนำ|บอก)',
        ]
        
        if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in high_confidence):
            return True
        
        # TIER 2: Medium confidence (meeting + investment context)
        meeting_terms = [r'เข้าพบ', r'มาพบ', r'พบ(?:\s+(?:กัน|คุณ|ท่าน))', r'วันนี้.*มา', r'ครั้งนี้.*มา']
        purpose_terms = [r'เพื่อ', r'อธิบาย', r'แนะนำ', r'บอก', r'เสนอ', r'นำเสนอ']
        investment_terms = [
            r'การลงทุน', r'ลงทุน', r'พอร์ต(?:โฟลิโอ)?', r'กองทุน', r'หุ้น', 
            r'ตราสาร', r'ผลิตภัณฑ์', r'พรอดัก', r'ธนาคาร', r'scb', r'บริการ'
        ]
        
        has_meeting = any(re.search(term, text_lower, re.IGNORECASE) for term in meeting_terms)
        has_purpose = any(re.search(term, text_lower, re.IGNORECASE) for term in purpose_terms)
        has_investment = any(re.search(term, text_lower, re.IGNORECASE) for term in investment_terms)
        
        if has_meeting and has_purpose and has_investment:
            return True
        
        # TIER 3: Catch remaining cases - broader net for the last few false negatives
        broad_purpose_indicators = [
            r'(?:วันนี้|ครั้งนี้).*(?:การลงทุน|ลงทุน)',
            r'(?:การลงทุน|ลงทุน).*(?:วันนี้|ครั้งนี้)',
            r'(?:อธิบาย|แนะนำ|บอก).*(?:เรื่อง|เกี่ยวกับ)',
            r'(?:เรื่อง|เกี่ยวกับ).*(?:การลงทุน|ลงทุน)',
            r'ที่.*(?:มา|เข้า).*(?:วันนี้|ครั้งนี้)',
        ]
        
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in broad_purpose_indicators)
    
    @staticmethod
    def _detect_benefit_info(text: str) -> bool:
        """
        Already at 100% recall - maintain it while trying to improve precision slightly
        Current: 100% recall, 51% precision, 42 false positives
        """
        text_lower = text.lower()
        
        # TIER 1: Very high confidence benefit patterns
        high_confidence = [
            r'ประโยชน์.*(?:คุณ|ลูกค้า|ท่าน|ได้รับ)',
            r'(?:คุณ|ลูกค้า|ท่าน).*(?:ได้รับ|จะได้).*ประโยชน์',
            r'ช่วยให้.*(?:คุณ|ลูกค้า|ท่าน).*(?:ได้|สามารถ)',
            r'(?:คุณ|ลูกค้า|ท่าน).*จะ.*(?:ดีขึ้น|ปรับปรุง|เหมาะสม)',
            
            # Investment-specific benefits
            r'ปรับ(?:สัดส่วน|ปรุง).*การลงทุน.*(?:ให้|เพื่อ|ช่วย)',
            r'การลงทุน.*(?:ดีขึ้น|เหมาะสม|ปรับปรุง).*(?:กับคุณ|สำหรับคุณ)',
            r'(?:ลด|จัดการ).*ความเสี่ยง.*(?:ของคุณ|สำหรับคุณ)',
            r'เพิ่ม.*ผลตอบแทน.*(?:ของคุณ|สำหรับคุณ)',
            
            # Expert guidance benefits  
            r'(?:แนะนำ|คำแนะนำ).*(?:ที่|เพื่อ).*(?:เหมาะสม|ดี).*(?:กับคุณ|สำหรับคุณ)',
            r'scbcio.*แนะนำ.*(?:เพื่อ|ให้|ช่วย)',
        ]
        
        if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in high_confidence):
            return True
        
        # TIER 2: Medium confidence (need multiple signals)
        benefit_words = [r'ช่วย', r'ประโยชน์', r'ดีขึ้น', r'ปรับ', r'เพิ่ม', r'แนะนำ']
        customer_words = [r'คุณ', r'ลูกค้า', r'ท่าน']
        investment_words = [r'การลงทุน', r'ลงทุน', r'พอร์ต', r'ความเสี่ยง', r'ผลตอบแทน']
        
        benefit_count = sum(1 for word in benefit_words if re.search(word, text_lower, re.IGNORECASE))
        has_customer = any(re.search(word, text_lower, re.IGNORECASE) for word in customer_words)
        has_investment = any(re.search(word, text_lower, re.IGNORECASE) for word in investment_words)
        
        # Need 2+ benefit words AND customer reference AND investment context
        return benefit_count >= 2 and has_customer and has_investment
    
    @staticmethod
    def _detect_interval_info(text: str) -> bool:
        """
        Improved interval detection based on specific criteria.
        
        แจ้งระยะเวลาที่ใช้ในการเข้าพบ
        
        Do:
        - ขอเวลา 1 ชั่วโมง สะดวกมั้ย
        - ขอเวลา 30 นาที สะดวกมั้ย
        
        Don't:  
        - ไม่ได้แจ้งระยะเวลาที่ใช้ในการเข้าพบเลย
        """
        
        text_lower = text.lower()
        
        # TIER 1: High confidence patterns - specific meeting time requests
        high_confidence_patterns = [
            # Direct time requests (exactly matching criteria)
            r'ขอเวลา\s*\d+\s*(?:ชั่วโมง|นาที)\s*(?:สะดวก|มั้ย|ไหม|ครับ|ค่ะ|คะ)',
            r'ขอเวลา\s*(?:ประมาณ\s*)?\d+\s*(?:ชั่วโมง|นาที)',
            
            # Meeting duration context
            r'(?:ใช้เวลา|ใช้เวลาประมาณ)\s*\d+\s*(?:ชั่วโมง|นาที|วัน)',
            r'(?:เวลา|ระยะเวลา|ช่วงเวลา)\s*(?:ประมาณ\s*)?\d+\s*(?:ชั่วโมง|นาที|วัน)',
            
            # Polite time requests
            r'(?:ขอ|จะขอ)\s*(?:เวลา|เวลาประมาณ)\s*\d+\s*(?:ชั่วโมง|นาที)',
            r'กี่\s*(?:ชั่วโมง|นาที)\s*(?:ครับ|ค่ะ|คะ|ดี|เหมาะสม)',
        ]
        
        # Check high confidence patterns first
        if any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in high_confidence_patterns):
            return True
        
        # TIER 2: Medium confidence - time mentions in meeting context
        meeting_context_keywords = [
            r'(?:เข้า|มา)*พบ', r'นัด(?:หมาย)?', r'ประชุม', r'คุย', r'พูดคุย', 
            r'อธิบาย', r'แนะนำ', r'นำเสนอ', r'ปรึกษา'
        ]
        
        time_patterns = [
            r'\d+\s*(?:ชั่วโมง|นาที|วัน)',
            r'(?:ระยะเวลา|ช่วงเวลา|ใช้เวลา)',
            r'กี่\s*(?:ชั่วโมง|นาที)',
        ]
        
        # Check if there's meeting context AND time mention
        has_meeting_context = any(re.search(keyword, text_lower, re.IGNORECASE) 
                                for keyword in meeting_context_keywords)
        has_time_mention = any(re.search(pattern, text_lower, re.IGNORECASE) 
                            for pattern in time_patterns)
        
        if has_meeting_context and has_time_mention:
            return True
        
        # TIER 3: Specific time request patterns (more conservative)
        specific_patterns = [
            r'ขอเวลา\s*\d+',  # Any "ขอเวลา" + number
            r'(?:ประมาณ|ราว|ราวๆ|ช่วง)\s*\d+\s*(?:ชั่วโมง|นาที)',  # Approximate time
            r'\d+\s*(?:ชั่วโมง|นาที)\s*(?:เพียงพอ|พอ|ดี|เหมาะสม)',  # Time adequacy
        ]
        
        return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in specific_patterns)

# Statistics tracking
class ServerStatistics:
    """Track server performance statistics."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.total_transcription_time = 0.0
        self.audio_files_processed = 0
        self.total_audio_duration = 0.0

    def record_request(self, success: bool, processing_time: float, transcription_time: float = 0.0, audio_duration: float = 0.0):
        """Record a request for statistics."""
        self.total_requests += 1
        self.total_processing_time += processing_time
        self.total_transcription_time += transcription_time
        
        if success:
            self.successful_requests += 1
            self.audio_files_processed += 1
            self.total_audio_duration += audio_duration
        else:
            self.failed_requests += 1

    @property
    def uptime_seconds(self) -> float:
        """Calculate server uptime in seconds."""
        return (datetime.now() - server_start_time).total_seconds()

# Global instances
asr_processor = None
server_stats = ServerStatistics()

# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global asr_processor
    
    # Startup
    logger.info("🚀 Multimodal LLM ASR API Server starting up...")
    logger.info(f"📊 Server configuration: {SERVER_CONFIG}")
    
    try:
        # Initialize ASR processor
        logger.info("🤖 Loading ASR model...")
        asr_processor = ASRProcessor()
        logger.info("✅ ASR model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ASR processor: {e}")
        # Don't raise here - let the server start but mark as not ready
        asr_processor = None
    
    logger.info("✅ Server startup complete")
    
    yield
    
    # Shutdown
    logger.info("🛑 Server shutting down...")
    
    if asr_processor:
        logger.info("🤖 Cleaning up ASR resources...")
        # Cleanup would go here if needed
    
    logger.info("✅ Server shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Multimodal LLM ASR API",
    description="Real ASR processing server using Pathumma Whisper model for Thai speech",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    if SERVER_CONFIG["enable_detailed_logging"]:
        logger.info(f"📥 {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    if SERVER_CONFIG["enable_detailed_logging"]:
        logger.info(f"📤 {request.method} {request.url} - "
                   f"Status: {response.status_code} - "
                   f"Time: {process_time:.3f}s")
    
    return response

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic server information."""
    return {
        "message": "Multimodal LLM ASR API Server",
        "version": "2.0.0",
        "status": "running",
        "model": SERVER_CONFIG["model_name"],
        "language": SERVER_CONFIG["language"],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy" if asr_processor and asr_processor.pipeline else "initializing",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=server_stats.uptime_seconds,
        model_loaded=asr_processor is not None and asr_processor.pipeline is not None,
        gpu_available=torch.cuda.is_available()
    )

@app.get("/stats", response_model=Dict[str, Any])
async def get_server_statistics():
    """Get server performance statistics."""
    return {
        "uptime_seconds": server_stats.uptime_seconds,
        "total_requests": server_stats.total_requests,
        "successful_requests": server_stats.successful_requests,
        "failed_requests": server_stats.failed_requests,
        "success_rate_percent": round((server_stats.successful_requests / max(server_stats.total_requests, 1)) * 100, 2),
        "audio_files_processed": server_stats.audio_files_processed,
        "total_audio_duration_seconds": round(server_stats.total_audio_duration, 2),
        "average_processing_time_seconds": round(server_stats.total_processing_time / max(server_stats.total_requests, 1), 3),
        "average_transcription_time_seconds": round(server_stats.total_transcription_time / max(server_stats.successful_requests, 1), 3),
        "requests_per_minute": round(server_stats.total_requests / (server_stats.uptime_seconds / 60), 2) if server_stats.uptime_seconds > 0 else 0,
        "model_info": {
            "name": SERVER_CONFIG["model_name"],
            "language": SERVER_CONFIG["language"],
            "device": "GPU" if SERVER_CONFIG["enable_gpu"] and torch.cuda.is_available() else "CPU",
            "chunk_length_ms": SERVER_CONFIG["chunk_length_ms"]
        }
    }

@app.post("/eval", response_model=AudioAnalysisResponse)
async def evaluate_audio(
    background_tasks: BackgroundTasks,
    voice_file: UploadFile = File(..., description="Audio file to analyze"),
    agent_data: str = Form(..., description="Agent information as JSON string")
):
    """
    Main evaluation endpoint for audio analysis with real ASR processing.
    """
    start_time = time.time()
    transcription_start_time = 0
    success = False
    temp_file_path = None
    
    try:
        # Validate file size
        file_size_mb = 0
        if hasattr(voice_file, 'size') and voice_file.size:
            file_size_mb = voice_file.size / (1024 * 1024)
            if file_size_mb > SERVER_CONFIG["max_file_size_mb"]:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File too large: {file_size_mb:.1f}MB. Max size: {SERVER_CONFIG['max_file_size_mb']}MB"
                )
        
        # Validate audio file
        if not voice_file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Validate file type
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        file_extension = Path(voice_file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format: {file_extension}. "
                       f"Supported formats: {', '.join(allowed_extensions)}"
            )
        
        # Parse agent data
        try:
            agent_data_dict = json.loads(agent_data)
            agent_info = AgentData(**agent_data_dict)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid agent data format: {str(e)}"
            )
        
        # Check if ASR processor is ready
        if not asr_processor or not asr_processor.pipeline:
            raise HTTPException(
                status_code=503, 
                detail="ASR model not ready. Please try again in a few moments."
            )
        
        logger.info(f"🔊 Processing audio: {voice_file.filename} "
                   f"({file_size_mb:.1f}MB) for agent: {agent_info.agent_fname} {agent_info.agent_lname}")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file_path = temp_file.name
            content = await voice_file.read()
            temp_file.write(content)
        
        # Get audio duration
        try:
            audio_segment = AudioSegment.from_file(temp_file_path)
            audio_duration_seconds = len(audio_segment) / 1000.0
        except:
            audio_duration_seconds = 0.0
        
        # Perform ASR transcription
        transcription_start_time = time.time()
        transcription = await asr_processor.transcribe_audio(temp_file_path)
        transcription_time = time.time() - transcription_start_time
        
        logger.info(f"📝 Transcription completed in {transcription_time:.2f}s: {len(transcription)} characters")
        
        # Analyze transcription content
        analysis_result = TextAnalyzer.analyze_speech_content(transcription, agent_info)
        
        # Create response
        response = AudioAnalysisResponse(
            **analysis_result,
            agent_name=f"{agent_info.agent_fname} {agent_info.agent_lname}",
            transcription=transcription,
            audio_duration_seconds=round(audio_duration_seconds, 2),
            processing_time_seconds=round(time.time() - start_time, 3),
            confidence_score=0.95,  # Could be enhanced with actual confidence from model
            timestamp=datetime.now().isoformat()
        )
        
        success = True
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Record statistics
        processing_time = time.time() - start_time
        transcription_time = time.time() - transcription_start_time if transcription_start_time > 0 else 0
        audio_duration = 0
        
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                audio_segment = AudioSegment.from_file(temp_file_path)
                audio_duration = len(audio_segment) / 1000.0
        except:
            pass
        
        server_stats.record_request(success, processing_time, transcription_time, audio_duration)
        
        # Schedule cleanup of temporary file
        if temp_file_path and SERVER_CONFIG["cleanup_temp_files"]:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)

def cleanup_temp_file(file_path: str):
    """Clean up temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"🧹 Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to cleanup temp file {file_path}: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.warning(f"⚠️ HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"💥 Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Main function to run the server
def main():
    """Run the FastAPI server."""
    print("🚀 Starting Multimodal LLM ASR API Server")
    print(f"📋 Configuration:")
    print(f"   Host: {SERVER_CONFIG['host']}")
    print(f"   Port: {SERVER_CONFIG['port']}")
    print(f"   Model: {SERVER_CONFIG['model_name']}")
    print(f"   Language: {SERVER_CONFIG['language']}")
    print(f"   GPU Enabled: {SERVER_CONFIG['enable_gpu']}")
    print(f"   Chunk Length: {SERVER_CONFIG['chunk_length_ms']}ms")
    print(f"   Max File Size: {SERVER_CONFIG['max_file_size_mb']}MB")
    print("\n📖 API Documentation:")
    print(f"   Swagger UI: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}/docs")
    print(f"   ReDoc: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}/redoc")
    print("\n🎯 Main Endpoint:")
    print(f"   POST /eval - Real ASR processing endpoint")
    print("\n💡 Usage with load test:")
    print(f"   locust -f loadtest_main.py --users 5 --spawn-rate 1 --run-time 5m --host http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print("\n⚠️  Note: Lower concurrency recommended for ASR processing")
    

if __name__ == "__main__":
    import uvicorn
    # Calculate optimal workers
    workers = 4  # Max 4 workers for ASR
    
    uvicorn.run(
        "api_rulebased:app",
        host="0.0.0.0",
        port=4000,
        workers=workers,
        loop="uvloop",  # Faster event loop
        log_level="info",
        access_log=True,
        use_colors=True
    )

# requirements.txt for the ASR server
"""
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
torch>=2.0.0
transformers>=4.35.0
pydub>=0.25.1
pandas>=2.0.0
tqdm>=4.65.0
"""

# docker-compose.yml for ASR server
"""
version: '3.8'
services:
  asr-api:
    build: .
    ports:
      - "4000:4000"
    environment:
      - HOST=0.0.0.0
      - PORT=4000
      - ENABLE_GPU=true
    volumes:
      - ./temp:/app/temp
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""

# Dockerfile for ASR server
"""
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create temp directory
RUN mkdir -p /app/temp

EXPOSE 4000

CMD ["python", "fastapi_asr_server.py"]
"""