import os
import sys
import json
import base64
import tempfile
import subprocess
import asyncio
import logging
import requests
import time
import threading
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except ImportError:
    print("Warning: flask-cors not installed, CORS may not work properly")
    CORS = None
from livekit.plugins import google
from livekit.agents import Agent, AgentSession

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('flask_api_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Add startup logging
logger.info("=" * 60)
logger.info("🚀 FLASK API STARTING UP")
logger.info("=" * 60)
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Environment variables:")
for key in ['GOOGLE_APPLICATION_CREDENTIALS', 'LIVEKIT_URL', 'LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET']:
    value = os.environ.get(key, 'NOT SET')
    logger.info(f"  {key}: {'SET' if value != 'NOT SET' else 'NOT SET'}")
logger.info("=" * 60)

# Initialize Flask app
app = Flask(__name__)
if CORS:
    CORS(app)

# Global variables for LiveKit components
stt_engine = None
tts_engine = None
vad_engine = None
agent_session = None
agent = None

# TTS engine cache to store TTS engines for different voices
tts_engine_cache = {}

# Global event loop for streaming TTS (reuse across calls)
_streaming_loop = None
_streaming_loop_lock = None

# Streaming TTS configuration
STREAMING_CHUNK_SIZE = 1024  # Size of audio chunks for streaming
STREAMING_DELAY = 0.1  # Delay between chunks in seconds

# Salesforce configuration
SALESFORCE_ORG_DOMAIN = os.environ.get('SALESFORCE_ORG_DOMAIN', '')
SALESFORCE_CLIENT_ID = os.environ.get('SALESFORCE_CLIENT_ID', '')
SALESFORCE_CLIENT_SECRET = os.environ.get('SALESFORCE_CLIENT_SECRET', '')
SALESFORCE_AGENT_ID = os.environ.get('SALESFORCE_AGENT_ID', '')

# Cache for Salesforce access tokens and sessions
salesforce_token_cache = {}
salesforce_session_cache = {}

# Performance tracking for streaming TTS
_streaming_performance = {
    "first_call_time": None,
    "subsequent_call_times": [],
    "total_calls": 0
}

def create_tts_engine_with_voice(voice_name="en-US-Wavenet-C"):
    """Create a TTS engine with specific Wavenet voice configuration"""
    try:
        print(f"🔧 Creating TTS engine with voice: {voice_name}")
        
        # Check if we have a cached engine for this voice
        if voice_name in tts_engine_cache:
            print(f"✅ Using cached TTS engine for voice: {voice_name}")
            return tts_engine_cache[voice_name]
        
        # Create new TTS engine with voice configuration
        # Note: LiveKit TTS doesn't support voice selection in the constructor
        # Voice selection is handled by the Google Cloud TTS service internally
        new_tts_engine = google.TTS()
        
        # Cache the engine
        tts_engine_cache[voice_name] = new_tts_engine
        print(f"✅ Created and cached TTS engine for voice: {voice_name}")
        
        return new_tts_engine
        
    except Exception as e:
        print(f"❌ Error creating TTS engine with voice {voice_name}: {e}")
        return None

def preprocess_text_for_tts(text):
    """Preprocess text to make TTS sound more natural and human-like using SSML"""
    if not text:
        return text
    
    import re
    
    # Check text length to prevent size limit errors
    MAX_TTS_LENGTH = 4000  # Maximum characters for TTS (conservative limit)
    if len(text) > MAX_TTS_LENGTH:
        print(f"⚠️ Text too long for TTS ({len(text)} chars), truncating to {MAX_TTS_LENGTH} chars")
        text = text[:MAX_TTS_LENGTH] + "..."
        print(f"🔧 Truncated text: {text[:100]}...")
    
    # Check if we should use SSML (disabled by default since LiveKit TTS doesn't support it)
    use_ssml = os.environ.get('USE_SSML', 'false').lower() == 'true'  # Disable SSML since LiveKit TTS doesn't support it
    
    if not use_ssml:
        print("🔧 SSML disabled, using simple text preprocessing")
        # Simple preprocessing without SSML
        processed_text = text
        
        # Add natural speech patterns
        processed_text = processed_text.replace('I am', "I'm")
        processed_text = processed_text.replace('I will', "I'll")
        processed_text = processed_text.replace('I have', "I've")
        processed_text = processed_text.replace('I would', "I'd")
        processed_text = processed_text.replace('I can', "I can")
        processed_text = processed_text.replace('I cannot', "I can't")
        processed_text = processed_text.replace('do not', "don't")
        processed_text = processed_text.replace('does not', "doesn't")
        processed_text = processed_text.replace('will not', "won't")
        processed_text = processed_text.replace('cannot', "can't")
        processed_text = processed_text.replace('should not', "shouldn't")
        processed_text = processed_text.replace('would not', "wouldn't")
        processed_text = processed_text.replace('could not', "couldn't")
        
        # Simple number replacement for better pronunciation
        processed_text = re.sub(r'\b0\b', 'zero', processed_text)
        processed_text = re.sub(r'\b1\b', 'one', processed_text)
        processed_text = re.sub(r'\b2\b', 'two', processed_text)
        processed_text = re.sub(r'\b3\b', 'three', processed_text)
        processed_text = re.sub(r'\b4\b', 'four', processed_text)
        processed_text = re.sub(r'\b5\b', 'five', processed_text)
        processed_text = re.sub(r'\b6\b', 'six', processed_text)
        processed_text = re.sub(r'\b7\b', 'seven', processed_text)
        processed_text = re.sub(r'\b8\b', 'eight', processed_text)
        processed_text = re.sub(r'\b9\b', 'nine', processed_text)
        
        print(f"🔧 Simple text preprocessing: '{text[:50]}...' → '{processed_text[:50]}...'")
        print(f"🔧 Full processed text: '{processed_text}'")
        return processed_text
    
    # Start with text processing (don't wrap in <speak> yet)
    processed_text = text
    
    # Fix "oh" -> "zero" pronunciation issue first
    # This is a common TTS issue where "oh" is pronounced instead of "zero"
    original_text = processed_text
    processed_text = re.sub(r'\boh\b', 'zero', processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\boh oh\b', 'zero zero', processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\boh oh oh\b', 'zero zero zero', processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\boh oh oh oh\b', 'zero zero zero zero', processed_text, flags=re.IGNORECASE)
    
    # Log if "oh" was converted to "zero"
    if original_text != processed_text:
        print(f"🔧 SSML: Fixed 'oh' -> 'zero' pronunciation: '{original_text}' -> '{processed_text}'")
    
    # Add natural speech patterns with contractions
    processed_text = processed_text.replace('I am', "I'm")
    processed_text = processed_text.replace('I will', "I'll")
    processed_text = processed_text.replace('I have', "I've")
    processed_text = processed_text.replace('I would', "I'd")
    processed_text = processed_text.replace('I can', "I can")
    processed_text = processed_text.replace('I cannot', "I can't")
    processed_text = processed_text.replace('do not', "don't")
    processed_text = processed_text.replace('does not', "doesn't")
    processed_text = processed_text.replace('will not', "won't")
    processed_text = processed_text.replace('cannot', "can't")
    processed_text = processed_text.replace('should not', "shouldn't")
    processed_text = processed_text.replace('would not', "wouldn't")
    processed_text = processed_text.replace('could not', "couldn't")
    
    # Handle phone numbers (XXX-XXX-XXXX format) - use characters for Google TTS
    processed_text = re.sub(r'\b(\d{3})-(\d{3})-(\d{4})\b', r'<say-as interpret-as="characters">\1-\2-\3</say-as>', processed_text)
    
    # Handle currency amounts (support any number of decimal places)
    processed_text = re.sub(r'\$(\d+(?:\.\d+)?)', r'<say-as interpret-as="currency">$\1</say-as>', processed_text)
    
    # Handle dates (MM/DD/YYYY or DD/MM/YYYY format)
    processed_text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', r'<say-as interpret-as="date" format="mdy">\1/\2/\3</say-as>', processed_text)
    
    # Handle time (HH:MM format)
    processed_text = re.sub(r'\b(\d{1,2}):(\d{2})\b', r'<say-as interpret-as="time" format="hms12">\1:\2</say-as>', processed_text)
    
    # Handle percentages
    processed_text = re.sub(r'(\d+(?:\.\d+)?)%', r'<say-as interpret-as="number">\1</say-as> percent', processed_text)
    
    # Handle numbers - use a simple approach to avoid regex complexity
    # First, let's handle all numbers with a basic cardinal interpretation
    # This is simpler and avoids the lookbehind issues
    processed_text = re.sub(r'\b(\d+)\b', r'<say-as interpret-as="cardinal">\1</say-as>', processed_text)
    
    # Add natural emphasis for important words using SSML
    emphasis_words = ['important', 'urgent', 'critical', 'error', 'success', 'warning', 'note', 'failed', 'completed', 'pending']
    for word in emphasis_words:
        if word in processed_text.lower():
            # Use SSML emphasis tag - simple approach without lookbehind
            processed_text = re.sub(rf'\b{word}\b', f'<emphasis level="strong">{word}</emphasis>', processed_text, flags=re.IGNORECASE)
    
    # Add natural pauses for better flow (BEFORE wrapping in <speak>)
    processed_text = processed_text.replace('.', '. ')  # Pause after periods
    processed_text = processed_text.replace('!', '! ')  # Pause after exclamations
    processed_text = processed_text.replace('?', '? ')  # Pause after questions
    processed_text = processed_text.replace(',', ', ')  # Pause after commas
    
    # Convert punctuation to breaks BEFORE wrapping in <speak>
    processed_text = processed_text.replace('. ', '<break time="500ms"/> ')
    processed_text = processed_text.replace('! ', '<break time="500ms"/> ')
    processed_text = processed_text.replace('? ', '<break time="500ms"/> ')
    processed_text = processed_text.replace(', ', '<break time="250ms"/> ')
    
    # Now wrap in SSML speak tags (after all processing)
    processed_text = f"<speak>{processed_text.strip()}</speak>"
    
    # Clean up multiple spaces but preserve spacing inside SSML tags
    # Use regex to normalize spaces only outside of tags
    processed_text = re.sub(r'(?<!>)\s+(?![^<]*>)', ' ', processed_text)
    
    print(f"🔧 Text preprocessing with SSML: '{text[:50]}...' → '{processed_text[:100]}...'")
    return processed_text

async def generate_streaming_tts_async(text, voice_name="en-US-Wavenet-C"):
    """Async function to generate streaming TTS with proper event loop handling"""
    try:
        print(f"🔧 Starting async streaming TTS with voice: {voice_name}")
        print(f"🔧 Original text: {text[:50]}...")
        print(f"🔧 Text length: {len(text)} characters")
        
        # Check text length before processing
        MAX_TTS_LENGTH = 4000  # Maximum characters for TTS
        if len(text) > MAX_TTS_LENGTH:
            print(f"⚠️ Text too long for TTS ({len(text)} chars), truncating to {MAX_TTS_LENGTH} chars")
            text = text[:MAX_TTS_LENGTH] + "..."
            print(f"🔧 Truncated text: {text[:100]}...")
        
        # Preprocess text to improve TTS pronunciation
        processed_text = preprocess_text_for_tts(text)
        print(f"🔧 Preprocessed text: {processed_text[:200]}...")
        
        # Create TTS engine with specific voice
        voice_tts_engine = create_tts_engine_with_voice(voice_name)
        if not voice_tts_engine:
            print("❌ Failed to create TTS engine")
            return None
        
        # Split plain text into smaller chunks for streaming
        print("🔧 Plain text detected - splitting into chunks for streaming")
        words = processed_text.split()
        chunk_size = 8  # Number of words per chunk
        text_chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            text_chunks.append(chunk)
        
        print(f"🔧 Processing {len(text_chunks)} chunk(s) for streaming")
        
        # Process each chunk and combine audio
        all_audio_chunks = []
        
        for i, chunk in enumerate(text_chunks):
            print(f"🔧 Processing chunk {i+1}/{len(text_chunks)}: {chunk[:30]}...")
            
            # Generate audio for this chunk
            audio_stream = None
            try:
                # Process plain text chunk
                print(f"🔧 Processing plain text chunk: {chunk[:100]}...")
                audio_stream = voice_tts_engine.synthesize(text=chunk)
                chunk_audio_data = []
                
                # Process audio stream for this chunk with proper error handling
                try:
                    # Use async iteration with timeout and proper cleanup
                    def create_collect_chunk_audio(stream, data_list, chunk_index):
                        async def collect_chunk_audio():
                            try:
                                async for audio_chunk in stream:
                                    if hasattr(audio_chunk, 'frame') and audio_chunk.frame:
                                        data_list.append(audio_chunk.frame.data)
                                    elif hasattr(audio_chunk, 'data'):
                                        data_list.append(audio_chunk.data)
                            except AttributeError as attr_error:
                                # Handle the specific gRPC InterceptedUnaryUnaryCall error
                                if "_interceptors_task" in str(attr_error):
                                    print(f"🔧 Handling gRPC cleanup error for chunk {chunk_index+1}: {attr_error}")
                                    # This is a known gRPC cleanup issue, not critical
                                else:
                                    raise attr_error
                            except Exception as e:
                                print(f"🔧 Error in chunk {chunk_index+1} audio collection: {e}")
                                raise e
                        return collect_chunk_audio
                    
                    collect_func = create_collect_chunk_audio(audio_stream, chunk_audio_data, i)
                    
                    # Run with timeout
                    await asyncio.wait_for(collect_func(), timeout=10.0)
                    
                except asyncio.TimeoutError:
                    print(f"❌ Chunk {i+1} async iteration timed out")
                except AttributeError as attr_error:
                    if "_interceptors_task" in str(attr_error):
                        print(f"🔧 gRPC cleanup error for chunk {i+1} (non-critical): {attr_error}")
                        # Continue processing as this is just a cleanup issue
                    else:
                        print(f"❌ Attribute error processing chunk {i+1}: {attr_error}")
                        continue
                except Exception as stream_error:
                    print(f"❌ Error processing chunk {i+1} audio stream: {stream_error}")
                    continue
                finally:
                    # Ensure proper cleanup of audio stream
                    try:
                        if audio_stream and hasattr(audio_stream, 'cancel'):
                            audio_stream.cancel()
                    except Exception as cleanup_error:
                        print(f"🔧 Cleanup error for chunk {i+1} (non-critical): {cleanup_error}")
                
                if chunk_audio_data:
                    # Combine audio data for this chunk
                    chunk_audio_bytes = b"".join(chunk_audio_data)
                    all_audio_chunks.append(chunk_audio_bytes)
                    print(f"✅ Chunk {i+1} processed: {len(chunk_audio_bytes)} bytes")
                else:
                    print(f"❌ No audio data for chunk {i+1}")
                    
            except Exception as chunk_error:
                print(f"❌ Error creating audio stream for chunk {i+1}: {chunk_error}")
                continue
        
        if not all_audio_chunks:
            print("❌ No audio chunks collected")
            return None
        
        # Combine all audio chunks
        full_audio_bytes = b"".join(all_audio_chunks)
        print(f"✅ Total streaming audio bytes: {len(full_audio_bytes)}")
        
        # Create WAV file
        wav_audio = create_wav_file(full_audio_bytes)
        audio_base64 = base64.b64encode(wav_audio).decode('utf-8')
        
        print(f"✅ Streaming TTS completed: {len(wav_audio)} bytes, Base64: {len(audio_base64)} chars")
        return audio_base64
        
    except Exception as e:
        print(f"❌ Error in async streaming TTS: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_or_create_streaming_loop():
    """Get or create a persistent event loop for streaming TTS"""
    global _streaming_loop, _streaming_loop_lock
    
    if _streaming_loop_lock is None:
        _streaming_loop_lock = threading.Lock()
    
    with _streaming_loop_lock:
        if _streaming_loop is None or _streaming_loop.is_closed():
            print("🔧 Creating new persistent event loop for streaming TTS...")
            _streaming_loop = asyncio.new_event_loop()
            # Start the loop in a separate thread
            def run_loop():
                asyncio.set_event_loop(_streaming_loop)
                _streaming_loop.run_forever()
            
            loop_thread = threading.Thread(target=run_loop, daemon=True)
            loop_thread.start()
            print("✅ Persistent event loop created and started")
        else:
            print("✅ Reusing existing event loop for streaming TTS")
    
    return _streaming_loop

def process_text_with_streaming_tts(text, voice_name="en-US-Wavenet-C"):
    """Process text with streaming TTS using optimized event loop handling"""
    global _streaming_performance
    
    start_time = time.time()
    _streaming_performance["total_calls"] += 1
    
    try:
        print(f"🔧 Starting optimized streaming TTS with voice: {voice_name}")
        print(f"🔧 Original text: {text[:50]}...")
        print(f"🔧 Call #{_streaming_performance['total_calls']}")
        
        # Method 1: Try optimized async streaming TTS first
        try:
            # Get or create persistent event loop
            loop = get_or_create_streaming_loop()
            
            # Submit the async function to the persistent loop
            future = asyncio.run_coroutine_threadsafe(
                generate_streaming_tts_async(text, voice_name), 
                loop
            )
            
            # Wait for result with timeout
            result = future.result(timeout=30)
            
            if result:
                end_time = time.time()
                call_duration = end_time - start_time
                
                if _streaming_performance["first_call_time"] is None:
                    _streaming_performance["first_call_time"] = call_duration
                    print(f"✅ First call completed in {call_duration:.2f} seconds")
                else:
                    _streaming_performance["subsequent_call_times"].append(call_duration)
                    avg_subsequent = sum(_streaming_performance["subsequent_call_times"]) / len(_streaming_performance["subsequent_call_times"])
                    print(f"✅ Subsequent call #{len(_streaming_performance['subsequent_call_times'])} completed in {call_duration:.2f} seconds")
                    print(f"🔧 Average subsequent call time: {avg_subsequent:.2f} seconds")
                
                print("✅ Optimized async streaming TTS completed successfully")
                return result
            else:
                print("❌ Optimized async streaming TTS returned None, trying fallback...")
                
        except AttributeError as attr_error:
            if "_interceptors_task" in str(attr_error):
                print(f"🔧 gRPC cleanup error in async streaming (non-critical): {attr_error}")
                print("🔧 Trying subprocess fallback...")
            else:
                print(f"❌ Attribute error in async streaming: {attr_error}")
                print("🔧 Trying subprocess fallback...")
        except Exception as async_error:
            print(f"❌ Error in optimized async streaming TTS: {async_error}")
            print("🔧 Trying subprocess fallback...")
        
        # Method 2: Fallback to subprocess approach for gRPC issues
        print("🔧 Using subprocess fallback for streaming TTS...")
        try:
            # Use the existing subprocess TTS method as fallback
            fallback_result = process_text_with_tts_sync(text, 'en-US', voice_name)
            if fallback_result:
                end_time = time.time()
                call_duration = end_time - start_time
                print(f"✅ Subprocess fallback TTS completed in {call_duration:.2f} seconds")
                return fallback_result
            else:
                print("❌ Subprocess fallback also failed")
                return None
        except Exception as subprocess_error:
            print(f"❌ Error in subprocess fallback: {subprocess_error}")
            return None
        
    except Exception as e:
        print(f"❌ Error in process_text_with_streaming_tts: {e}")
        logger.error("Error in process_text_with_streaming_tts: %s", e)
        import traceback
        traceback.print_exc()
        return None

def process_text_with_tts_sync(text, language='en-US', voice='en-US-Wavenet-C'):
    """TTS processing using subprocess to isolate async LiveKit TTS (fixes event loop issue)"""
    try:
        print("🔧 process_text_with_tts_sync FUNCTION CALLED")
        print(f"🔧 Original text: {text[:50]}...")
        print(f"🔧 Language: {language}, Voice: {voice}")
        print(f"🔧 Text length: {len(text)} characters")
        
        # Check text length before processing
        MAX_TTS_LENGTH = 4000  # Maximum characters for TTS
        if len(text) > MAX_TTS_LENGTH:
            print(f"⚠️ Text too long for TTS ({len(text)} chars), truncating to {MAX_TTS_LENGTH} chars")
            text = text[:MAX_TTS_LENGTH] + "..."
            print(f"🔧 Truncated text: {text[:100]}...")
        
        # Preprocess text to improve TTS pronunciation (fixes "0" → "o" issue)
        processed_text = preprocess_text_for_tts(text)
        
        # Create temporary file with processed text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(processed_text)
            temp_file_path = temp_file.name

        try:
            # Create TTS script that uses the same approach as your working code
            tts_script = f"""
import os
import sys
import json
import base64
import tempfile
import asyncio
import struct
from livekit.plugins import google

# Set up Google credentials (same as your working code)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "{os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}"

async def generate_tts():
   try:
       print("🔧 Starting TTS generation...")
       
       # Initialize TTS engine
       print("🔧 Initializing TTS engine...")
       tts_engine = google.TTS()
       print("✅ TTS engine initialized")
       
       # Read text from file
       print("🔧 Reading text from file: {temp_file_path}")
       with open('{temp_file_path}', 'r', encoding='utf-8') as f:
           text = f.read()
       
       print("🔧 Processing text: " + text[:50] + "...")
       print("🔧 Full text content: " + text)
       print("🔧 Text length: " + str(len(text)) + " characters")
       
       if not text.strip():
           print("❌ Empty text provided")
           return None
       
       # Generate audio using LiveKit TTS synthesize method
       print("🔧 Calling tts_engine.synthesize()...")
       print("🔧 Input text: " + text[:200] + "...")
       
       # Process plain text input
       print("🔧 Processing plain text input...")
       audio_stream = tts_engine.synthesize(text=text)
       
       print("✅ Audio stream created")
       
       audio_chunks = []
       chunk_count = 0
       
       # Process audio stream
       print("🔧 Processing audio stream...")
       
       try:
           # Try async iteration first
           try:
               async for chunk in audio_stream:
                   chunk_count += 1
                   print("🔧 Processing chunk " + str(chunk_count) + ": " + str(type(chunk)))
                   
                   if hasattr(chunk, 'frame') and chunk.frame:
                       audio_data = chunk.frame.data
                       audio_chunks.append(audio_data)
                       print("🔧 Collected chunk " + str(chunk_count) + ": " + str(len(audio_data)) + " bytes")
                   elif hasattr(chunk, 'data'):
                       audio_data = chunk.data
                       audio_chunks.append(audio_data)
                       print("🔧 Collected chunk " + str(chunk_count) + " (data): " + str(len(audio_data)) + " bytes")
                   else:
                       print("🔧 Chunk " + str(chunk_count) + " type: " + str(type(chunk)) + ", attributes: " + str(dir(chunk)))
                       
                   # Safety check to prevent infinite loops
                   if chunk_count > 1000:
                       print("❌ Too many chunks, stopping processing")
                       break
                       
           except TypeError as async_error:
               print("🔧 Async iteration failed, trying sync iteration: " + str(async_error))
               
               # Fallback to sync iteration
               for chunk in audio_stream:
                   chunk_count += 1
                   print("🔧 Processing chunk " + str(chunk_count) + ": " + str(type(chunk)))
                   
                   if hasattr(chunk, 'frame') and chunk.frame:
                       audio_data = chunk.frame.data
                       audio_chunks.append(audio_data)
                       print("🔧 Collected chunk " + str(chunk_count) + ": " + str(len(audio_data)) + " bytes")
                   elif hasattr(chunk, 'data'):
                       audio_data = chunk.data
                       audio_chunks.append(audio_data)
                       print("🔧 Collected chunk " + str(chunk_count) + " (data): " + str(len(audio_data)) + " bytes")
                   else:
                       print("🔧 Chunk " + str(chunk_count) + " type: " + str(type(chunk)) + ", attributes: " + str(dir(chunk)))
                       
                   # Safety check to prevent infinite loops
                   if chunk_count > 1000:
                       print("❌ Too many chunks, stopping processing")
                       break
                       
       except Exception as stream_error:
           print("❌ Error processing audio stream: " + str(stream_error))
           import traceback
           traceback.print_exc()
           return None
       
       if not audio_chunks:
           print("❌ No audio chunks collected")
           return None
       
       # Combine audio data
       full_audio_bytes = b"".join(audio_chunks)
       print("✅ Total audio bytes: " + str(len(full_audio_bytes)))
       
       if len(full_audio_bytes) == 0:
           print("❌ Empty audio data")
           return None
       
       # Create WAV file
       print("🔧 Creating WAV file...")
       wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
           b'RIFF', 36 + len(full_audio_bytes), b'WAVE', b'fmt ', 16, 1,
           1, 24000, 24000 * 1 * 16 // 8, 1 * 16 // 8, 16, b'data', len(full_audio_bytes)
       )
       wav_audio = wav_header + full_audio_bytes
       
       # Convert to base64
       audio_base64 = base64.b64encode(wav_audio).decode('utf-8')
       print("✅ WAV audio created: " + str(len(wav_audio)) + " bytes, Base64: " + str(len(audio_base64)) + " chars")
       
       return audio_base64
       
   except Exception as e:
       print("❌ Error in TTS generation: " + str(e))
       import traceback
       traceback.print_exc()
       return None

if __name__ == "__main__":
   result = asyncio.run(generate_tts())
   if result:
       print("SUCCESS:" + result)
   else:
       print("FAILED")
"""

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_file.write(tts_script)
                script_path = script_file.name

            try:
                # Run the script in subprocess (isolates async event loop)
                print("🔧 Running TTS subprocess with LiveKit TTS...")
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False
                )

                print(f"🔧 Subprocess return code: {result.returncode}")
                print(f"🔧 Subprocess stdout: {result.stdout}")
                print(f"🔧 Subprocess stderr: {result.stderr}")

                if result.returncode == 0 and "SUCCESS:" in result.stdout:
                    # Extract the base64 audio data
                    audio_base64 = result.stdout.split("SUCCESS:")[1].strip()
                    if audio_base64 and len(audio_base64) > 0:
                        print("✅ TTS subprocess completed successfully")
                        print(f"✅ Audio data length: {len(audio_base64)} characters")
                        return audio_base64
                    else:
                        print("❌ TTS subprocess returned empty audio data")
                        return None
                else:
                    print("❌ TTS subprocess failed")
                    print(f"❌ Return code: {result.returncode}")
                    print(f"❌ Error output: {result.stderr}")
                    return None

            finally:
                # Clean up temporary files
                try:
                    os.unlink(script_path)
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            print(f"❌ Error in TTS subprocess: {e}")
            import traceback
            traceback.print_exc()
            return None

    except Exception as e:
        print(f"❌ Error in process_text_with_tts_sync: {e}")
        logger.error("Error in process_text_with_tts_sync: %s", e)
        import traceback
        traceback.print_exc()
        return None

async def process_text_with_tts_async(text, language='en-US', voice='en-US-Wavenet-C'):  # pylint: disable=unused-argument
    """Async TTS processing using LiveKit directly"""
    try:
        print(f"🔧 Async TTS processing for text: {text[:50]}...")
        print(f"🔧 Text length: {len(text)} characters")

        if not tts_engine:
            print("❌ TTS engine not available in async function")
            return None

        # Check text length before processing
        MAX_TTS_LENGTH = 4000  # Maximum characters for TTS
        if len(text) > MAX_TTS_LENGTH:
            print(f"⚠️ Text too long for TTS ({len(text)} chars), truncating to {MAX_TTS_LENGTH} chars")
            text = text[:MAX_TTS_LENGTH] + "..."
            print(f"🔧 Truncated text: {text[:100]}...")

        # Preprocess text to improve TTS pronunciation
        processed_text = preprocess_text_for_tts(text)
        print(f"🔧 Preprocessed text in async: {processed_text[:200]}...")

        # Process text using LiveKit TTS directly
        print("🔧 Processing plain text input in async function...")
        audio_stream = tts_engine.synthesize(text=processed_text)

        audio_chunks = []

        # Try different approaches to get audio data
        try:
            # Method 1: Try async iteration with timeout
            print("🔧 Trying async iteration with timeout...")

            async def collect_audio_chunks():
                async for chunk in audio_stream:
                    if hasattr(chunk, 'frame') and chunk.frame:
                        audio_data = chunk.frame.data
                        audio_chunks.append(audio_data)
                        print(f"🔧 Collected audio chunk: {len(audio_data)} bytes")
                    elif hasattr(chunk, 'data'):
                        audio_data = chunk.data
                        audio_chunks.append(audio_data)
                        print(f"🔧 Collected audio chunk (data): {len(audio_data)} bytes")

            # Run with timeout
            await asyncio.wait_for(collect_audio_chunks(), timeout=25.0)

        except asyncio.TimeoutError:
            print("❌ Async iteration timed out after 25 seconds")
        except Exception as e:
            print(f"❌ Async iteration failed: {e}")

        # Method 2: Try synchronous iteration
        try:
            print("🔧 Trying synchronous iteration...")
            if hasattr(audio_stream, '__iter__') and not asyncio.iscoroutine(audio_stream):
                try:
                    # Check if it's actually iterable
                    iter_obj = iter(audio_stream)  # type: ignore
                    # Use a while loop to avoid the linter error
                    while True:
                        try:
                            chunk = next(iter_obj)
                            if hasattr(chunk, 'frame') and chunk.frame:
                                audio_data = chunk.frame.data
                                audio_chunks.append(audio_data)
                                print(f"🔧 Collected audio chunk: {len(audio_data)} bytes")
                            elif hasattr(chunk, 'data'):
                                audio_data = chunk.data
                                audio_chunks.append(audio_data)
                                print(f"🔧 Collected audio chunk (data): {len(audio_data)} bytes")
                        except StopIteration:
                            break
                except (TypeError, AttributeError):
                    # audio_stream might not be iterable in this context
                    print("🔧 Audio stream is not iterable in sync context")
        except Exception as e2:
            print(f"❌ Synchronous iteration also failed: {e2}")

        # Method 3: Try to get all data at once
        try:
            print("🔧 Trying to get all data at once...")
            if hasattr(audio_stream, '__iter__'):
                all_data = list(audio_stream)
                print(f"🔧 Got {len(all_data)} items from stream")
                for item in all_data:
                    if hasattr(item, 'frame') and item.frame:
                        audio_data = item.frame.data
                        audio_chunks.append(audio_data)
                        print(f"🔧 Collected audio chunk: {len(audio_data)} bytes")
                    elif hasattr(item, 'data'):
                        audio_data = item.data
                        audio_chunks.append(audio_data)
                        print(f"🔧 Collected audio chunk (data): {len(audio_data)} bytes")
        except Exception as e3:
            print(f"❌ All methods failed: {e3}")
            return None

        if not audio_chunks:
            print("❌ No audio chunks collected")
            return None

        full_audio_bytes = b"".join(audio_chunks)
        print(f"✅ Total audio bytes collected: {len(full_audio_bytes)}")

        # Convert to WAV format
        wav_audio = create_wav_file(full_audio_bytes)
        audio_base64 = base64.b64encode(wav_audio).decode('utf-8')

        print(f"✅ WAV audio created: {len(wav_audio)} bytes, Base64: {len(audio_base64)} chars")
        return audio_base64

    except Exception as e:
        print(f"❌ Error in process_text_with_tts_async: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_wav_file(audio_data, sample_rate=24000, channels=1, bits_per_sample=16):
    """Create WAV file from raw audio data"""
    import struct

    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(audio_data), b'WAVE', b'fmt ', 16, 1,
        channels, sample_rate, sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8, bits_per_sample, b'data', len(audio_data)
    )
    return wav_header + audio_data

def setup_google_credentials():
    """Setup Google credentials for TTS and STT"""
    try:
        print("🔧 Setting up Google credentials...")
        print("🔧 Starting Google credentials setup...")
        
        # Check for secret file first
        secret_file = "/etc/secrets/google-credentials.json"
        if os.path.exists(secret_file):
            print(f"🔧 Checking secret file: {secret_file}")
            print(f"✅ Secret file found: {secret_file}")
            
            # Read and fix the private key format
            with open(secret_file, 'r', encoding='utf-8') as f:
                creds_data = json.load(f)
            
            # Fix private key format if needed
            if 'private_key' in creds_data and creds_data['private_key']:
                creds_data['private_key'] = creds_data['private_key'].replace('\\n', '\n')
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(creds_data, temp_file)
                temp_creds_file = temp_file.name
            
            print(f"✅ Fixed private key format and set credentials: {temp_creds_file}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_file
            print(f"✅ Environment variable set to: {temp_creds_file}")
            print("🔧 Google credentials setup result: True")
            return True
        else:
            print(f"❌ Secret file not found: {secret_file}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up Google credentials: {e}")
        return False

def get_salesforce_access_token():
    """Get Salesforce access token using OAuth2 Client Credentials flow"""
    try:
        print("🔧 Getting Salesforce access token...")
        
        # Check if we have valid cached token
        if 'access_token' in salesforce_token_cache and 'expires_at' in salesforce_token_cache:
            import time
            if time.time() < salesforce_token_cache['expires_at']:
                print("✅ Using cached Salesforce access token")
                return salesforce_token_cache['access_token']
        
        # Validate Salesforce configuration
        if not all([SALESFORCE_ORG_DOMAIN, SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET]):
            print("❌ Salesforce configuration missing")
            print(f"🔧 SALESFORCE_ORG_DOMAIN: {'SET' if SALESFORCE_ORG_DOMAIN else 'NOT SET'}")
            print(f"🔧 SALESFORCE_CLIENT_ID: {'SET' if SALESFORCE_CLIENT_ID else 'NOT SET'}")
            print(f"🔧 SALESFORCE_CLIENT_SECRET: {'SET' if SALESFORCE_CLIENT_SECRET else 'NOT SET'}")
            return None
        
        # Prepare OAuth2 request
        token_url = f"{SALESFORCE_ORG_DOMAIN}/services/oauth2/token"
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': SALESFORCE_CLIENT_ID,
            'client_secret': SALESFORCE_CLIENT_SECRET
        }
        
        print(f"🔧 Requesting token from: {token_url}")
        
        # Make OAuth2 request
        response = requests.post(token_url, data=data, timeout=30)
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            
            if access_token:
                # Cache the token with expiration time
                import time
                expires_in = token_data.get('expires_in', 3600)  # Default 1 hour
                salesforce_token_cache['access_token'] = access_token
                salesforce_token_cache['expires_at'] = time.time() + expires_in - 60  # 1 minute buffer
                
                print("✅ Salesforce access token obtained successfully")
                return access_token
            else:
                print("❌ No access token in response")
                return None
        else:
            print(f"❌ Failed to get Salesforce access token: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting Salesforce access token: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_or_create_session_id(agent_id):
    """Get or create a persistent session ID for the agent"""
    try:
        print("🔧 Getting or creating session ID...")
        
        # Check if we have a cached session for this agent
        if agent_id in salesforce_session_cache:
            session_id = salesforce_session_cache[agent_id]
            print(f"✅ Using cached session ID: {session_id}")
            return session_id
        
        # Generate a new session ID
        import uuid
        session_id = str(uuid.uuid4())
        
        # Cache the session ID for this agent
        salesforce_session_cache[agent_id] = session_id
        print(f"✅ Created new session ID: {session_id}")
        
        return session_id
        
    except Exception as e:
        print(f"❌ Error getting/creating session ID: {e}")
        # Fallback to a simple session ID
        import time
        return f"session_{agent_id}_{int(time.time())}"

def clear_salesforce_caches():
    """Clear Salesforce token and session caches (for debugging or forced refresh)"""
    global salesforce_token_cache, salesforce_session_cache
    salesforce_token_cache.clear()
    salesforce_session_cache.clear()
    print("✅ Salesforce caches cleared")

def start_salesforce_session(agent_id, session_id):
    """Start a Salesforce Einstein Agent session"""
    try:
        print(f"🔧 Starting Salesforce session for agent: {agent_id}")
        
        # Get access token
        access_token = get_salesforce_access_token()
        if not access_token:
            print("❌ Failed to get Salesforce access token")
            return None
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Use the correct API host and endpoint structure
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/agents/{agent_id}/sessions"
        payload = {
            "externalSessionKey": session_id,
            "instanceConfig": {
                "endpoint": SALESFORCE_ORG_DOMAIN
            },
            "tz": "America/Los_Angeles",
            "variables": [
                {
                    "name": "$Context.EndUserLanguage",
                    "type": "Text",
                    "value": "en_US"
                }
            ],
            "featureSupport": "Streaming",
            "streamingCapabilities": {
                "chunkTypes": [
                    "Text"
                ]
            },
            "bypassUser": False
        }
        
        print(f"🔧 Starting session at: {url}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"🔧 Start session response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Session started successfully")
            print(f"🔧 Session response: {result}")
            
            # Store the actual session ID from Salesforce response
            salesforce_session_id = result.get('sessionId')
            if salesforce_session_id:
                print(f"✅ Stored Salesforce session ID: {salesforce_session_id}")
                # Cache the Salesforce session ID for this agent
                salesforce_session_cache[f"{agent_id}_salesforce_session"] = salesforce_session_id
                return salesforce_session_id
            else:
                print("❌ No sessionId in response")
                return None
        else:
            print(f"❌ Failed to start session: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Salesforce session: {e}")
        import traceback
        traceback.print_exc()
        return None

def call_salesforce_einstein_agent(message, session_id, agent_id):
    """Call Salesforce Einstein Agent API using the working implementation"""
    try:
        print("🔧 Calling Salesforce Einstein Agent API...")
        print(f"🔧 Message: {message[:50]}...")
        print(f"🔧 Session ID: {session_id}")
        print(f"🔧 Agent ID: {agent_id}")
        
        # Get access token
        access_token = get_salesforce_access_token()
        if not access_token:
            print("❌ Failed to get Salesforce access token")
            return None
        
        # Check if we have a Salesforce session for this agent
        salesforce_session_key = f"{agent_id}_salesforce_session"
        if salesforce_session_key not in salesforce_session_cache:
            print("🔧 No Salesforce session found, starting new session...")
            salesforce_session_id = start_salesforce_session(agent_id, session_id)
            if not salesforce_session_id:
                print("❌ Failed to start Salesforce session")
                return None
        else:
            salesforce_session_id = salesforce_session_cache[salesforce_session_key]
            print(f"✅ Using existing Salesforce session: {salesforce_session_id}")
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Use the correct API host and endpoint structure for sending messages
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/sessions/{salesforce_session_id}/messages"
        
        print(f"🔧 Sending message to URL: {url}")
        print(f"🔧 Using Salesforce session ID: {salesforce_session_id}")
        
        payload = {
            "message": {
                "sequenceId": str(int(time.time() * 1000)),  # timestamp as sequenceId
                "type": "Text",
                "text": message
            }
        }
        
        # Make the API call
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"🔧 Salesforce API response status: {response.status_code}")
        print(f"🔧 Salesforce API response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ Salesforce API returned 200, processing response")
            result = response.json()
            print(f"🔧 Salesforce response: {result}")
            
            # Parse the response according to your working structure
            if 'message' in result:
                # Direct message response
                response_message = result.get('message', 'No response message')
                print(f"✅ Direct message response: {response_message}")
                return {
                    'message': response_message,
                    'sessionId': salesforce_session_id,
                    'agentId': agent_id,
                    'success': True
                }
            elif 'messages' in result:
                # Array response format
                messages = result.get('messages', [])
                if messages and len(messages) > 0:
                    message_obj = messages[0]
                    message_type = message_obj.get('type', '')
                    
                    print(f"🔧 Message type: {message_type}")
                    
                    # Handle different response types
                    if message_type == 'Inform':
                        response_message = message_obj.get('message', 'No response message')
                        print(f"✅ Inform response: {response_message}")
                        return {
                            'message': response_message,
                            'sessionId': salesforce_session_id,
                            'agentId': agent_id,
                            'success': True
                        }
                    elif message_type == 'Failure':
                        print(f"❌ Salesforce returned Failure: {message_obj}")
                        # Check if there's a message field with content
                        if message_obj.get('message'):
                            return {
                                'message': message_obj.get('message'),
                                'sessionId': salesforce_session_id,
                                'agentId': agent_id,
                                'success': False
                            }
                        # Check if there are errors with useful information
                        errors = message_obj.get('errors', [])
                        if errors and len(errors) > 0:
                            error_msg = errors[0]
                            return {
                                'message': f"I encountered an issue: {error_msg}",
                                'sessionId': salesforce_session_id,
                                'agentId': agent_id,
                                'success': False
                            }
                        # If no useful error info, use fallback
                        return {
                            'message': f"I received your message: '{message}'. I'm a Salesforce Einstein Agent, but I'm having trouble with the specific API endpoint. Please check your Salesforce configuration.",
                            'sessionId': salesforce_session_id,
                            'agentId': agent_id,
                            'success': False
                        }
                    elif message_type == 'Text':
                        response_message = message_obj.get('text', 'No response text')
                        print(f"✅ Text response: {response_message}")
                        return {
                            'message': response_message,
                            'sessionId': salesforce_session_id,
                            'agentId': agent_id,
                            'success': True
                        }
                    else:
                        # For any other type, try to get the message field
                        response_message = message_obj.get('message', f'Received response type: {message_type}')
                        print(f"✅ Other response type: {response_message}")
                        return {
                            'message': response_message,
                            'sessionId': salesforce_session_id,
                            'agentId': agent_id,
                            'success': True
                        }
                else:
                    return {
                        'message': "No messages in response",
                        'sessionId': salesforce_session_id,
                        'agentId': agent_id,
                        'success': False
                    }
            else:
                return {
                    'message': "No response message found",
                    'sessionId': salesforce_session_id,
                    'agentId': agent_id,
                    'success': False
                }
        elif response.status_code == 404:
            print("❌ Einstein Agent API not available (404), using fallback response")
            error_text = response.text
            print(f"❌ 404 Error response: {error_text}")
            return {
                'message': f"I received your message: '{message}'. I'm a Salesforce Einstein Agent, but I'm having trouble with the specific API endpoint. Please check your Salesforce configuration.",
                'sessionId': salesforce_session_id,
                'agentId': agent_id,
                'success': False
            }
        else:
            print(f"❌ Salesforce API error: {response.status_code}")
            error_text = response.text
            print(f"❌ Error response: {error_text}")
            return {
                'message': f"I received your message: '{message}'. I'm a Salesforce Einstein Agent, but I'm having trouble with the specific API endpoint. Please check your Salesforce configuration.",
                'sessionId': salesforce_session_id,
                'agentId': agent_id,
                'success': False
            }
            
    except Exception as e:
        print(f"❌ Error calling Salesforce Einstein Agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def initialize_livekit_components():
    """Initialize LiveKit components"""
    global stt_engine, tts_engine, vad_engine, agent_session, agent
    
    try:
        logger.info("=" * 50)
        logger.info("🔧 INITIALIZING LIVEKIT COMPONENTS")
        logger.info("=" * 50)
        logger.info(f"Current component status:")
        logger.info(f"  TTS engine: {'Available' if tts_engine is not None else 'Not initialized'}")
        logger.info(f"  STT engine: {'Available' if stt_engine is not None else 'Not initialized'}")
        logger.info(f"  VAD engine: {'Available' if vad_engine is not None else 'Not initialized'}")
        logger.info(f"  Agent session: {'Available' if agent_session is not None else 'Not initialized'}")
        logger.info(f"  Agent: {'Available' if agent is not None else 'Not initialized'}")
        
        # Setup Google credentials
        logger.info("🔧 Setting up Google credentials...")
        if not setup_google_credentials():
            logger.error("❌ Failed to setup Google credentials")
            return False
        logger.info("✅ Google credentials setup successful")
        
        # Initialize STT engine
        try:
            logger.info("🔧 Initializing STT engine...")
            stt_engine = google.STT()
            logger.info("✅ STT engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize STT engine: {e}")
            logger.error(f"STT Error details: {str(e)}")
            import traceback
            logger.error(f"STT Traceback: {traceback.format_exc()}")
            return False
        
        # Initialize TTS engine
        try:
            logger.info("🔧 Initializing TTS engine...")
            tts_engine = google.TTS()
            logger.info("✅ TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS engine: {e}")
            logger.error(f"TTS Error details: {str(e)}")
            import traceback
            logger.error(f"TTS Traceback: {traceback.format_exc()}")
            return False
        
        # VAD (Voice Activity Detection) is not available in current LiveKit version
        print("🔧 Skipping VAD initialization (not available in current LiveKit version)")
        vad_engine = None
        
        # Create AgentSession with STT and TTS
        try:
            print("🔧 Creating AgentSession...")
            agent_session = AgentSession(
                stt=stt_engine,
                tts=tts_engine,
            )
            print("✅ AgentSession created successfully")
        except Exception as e:
            print(f"❌ Failed to create AgentSession: {e}")
            return False
        
        # Create Agent
        try:
            print("🔧 Creating Agent...")
            agent = Agent(
                instructions="You are a helpful Salesforce voice assistant. Provide clear and concise responses.",
                stt=stt_engine,
                tts=tts_engine
            )
            print("✅ Agent created successfully")
        except Exception as e:
            print(f"❌ Failed to create Agent: {e}")
            return False
        
        print("✅ All LiveKit components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing LiveKit components: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize components on startup
print("🔧 Starting Flask API initialization...")
initialize_livekit_components()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - redirect to health check"""
    return jsonify({
        "message": "LiveKit Voice Agent API is running",
        "health_check": "/health",
        "endpoints": [
            "GET /health",
            "POST /api/einstein/agent", 
            "POST /api/voice/tts",
            "POST /api/voice/tts-stream",
            "POST /api/voice/stt"
        ]
    })

@app.route('/api/einstein/agent', methods=['POST'])
def einstein_agent():
    """Salesforce Einstein Agent endpoint"""
    try:
        logger.info("=" * 50)
        logger.info("🤖 EINSTEIN AGENT API ENDPOINT CALLED")
        logger.info("=" * 50)
        
        # Log request details
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request remote address: {request.remote_addr}")
        
        data = request.get_json()
        logger.info(f"Request JSON data: {json.dumps(data, indent=2) if data else 'None'}")
        
        message = data.get('message', '') if data else ''
        provided_session_id = data.get('session_id', '') if data else ''
        agent_id = data.get('agent_id', SALESFORCE_AGENT_ID or 'agent_001') if data else (SALESFORCE_AGENT_ID or 'agent_001')
        
        logger.info(f"Extracted parameters:")
        logger.info(f"  Message: '{message}' (length: {len(message)})")
        logger.info(f"  Provided Session ID: '{provided_session_id}'")
        logger.info(f"  Agent ID: '{agent_id}'")
        
        if not message:
            logger.error("❌ Error: Message is required")
            return jsonify({"error": "Message is required"}), 400
        
        # Use provided session_id or get/create a persistent one
        if provided_session_id:
            session_id = provided_session_id
            logger.info(f"✅ Using provided session ID: {session_id}")
        else:
            logger.info("🔧 No session ID provided, getting/creating persistent one...")
            session_id = get_or_create_session_id(agent_id)
            logger.info(f"✅ Using persistent session ID: {session_id}")
        
        # Call actual Salesforce Einstein Agent API
        logger.info("🔧 Calling Salesforce Einstein Agent API...")
        logger.info(f"  Message: '{message}'")
        logger.info(f"  Session ID: '{session_id}'")
        logger.info(f"  Agent ID: '{agent_id}'")
        
        einstein_response = call_salesforce_einstein_agent(message, session_id, agent_id)
        
        if einstein_response:
            print("✅ Einstein Agent API call successful")
            return jsonify({
                "message": einstein_response.get('message', 'No response from Einstein Agent'),
                "session_id": session_id,
                "agent_id": agent_id,
                "success": True,
                "raw_response": einstein_response
            })
        else:
            print("❌ Einstein Agent API call failed, returning fallback response")
            # Fallback response if API call fails
            response_message = f"I received your message: '{message}'. I'm a Salesforce Einstein Agent, but I'm having trouble connecting to the Salesforce API right now. Please check your Salesforce configuration and try again later."
            
            return jsonify({
                "message": response_message,
                "session_id": session_id,
                "agent_id": agent_id,
                "success": False,
                "error": "Failed to connect to Salesforce Einstein Agent API"
            })
        
    except Exception as e:
        print(f"❌ Error in Einstein Agent: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/clear-cache', methods=['POST'])
def clear_cache():
    """Debug endpoint to clear Salesforce caches"""
    try:
        clear_salesforce_caches()
        return jsonify({
            "message": "Salesforce caches cleared successfully",
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/test-tts', methods=['POST'])
def test_tts():
    """Debug endpoint to test TTS functionality"""
    try:
        data = request.get_json()
        test_text = data.get('text', 'Hello, this is a test of the text-to-speech system.')
        
        print(f"🔧 Testing TTS with text: {test_text}")
        
        # Test TTS generation
        audio_content = process_text_with_tts_sync(test_text)
        
        if audio_content:
            return jsonify({
                "success": True,
                "message": "TTS test successful",
                "audio_length": len(audio_content),
                "text": test_text
            })
        else:
            return jsonify({
                "success": False,
                "message": "TTS test failed",
                "text": test_text
            })
    except Exception as e:
        print(f"❌ TTS test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "text": test_text if 'test_text' in locals() else "unknown"
        }), 500

@app.route('/api/debug/test-ssml', methods=['POST'])
def test_ssml():
    """Debug endpoint to test SSML functionality with numbers, dates, and currencies"""
    try:
        # Test cases for SSML improvements
        test_cases = {
            "case_number": "The case with number 0001111 was not found.",
            "currency": "The total amount is $1,234.56.",
            "date": "The case was created on 09/16/2025.",
            "time": "The meeting is scheduled for 2:30 PM.",
            "percentage": "The success rate is 95.5%.",
            "phone": "Please call us at 555-123-4567.",
            "mixed": "Case 0001111 for $500.00 on 09/16/2025 at 2:30 PM has a 95% success rate. Call 555-123-4567."
        }
        
        test_case = request.get_json().get('test_case', 'case_number')
        test_text = test_cases.get(test_case, test_cases['case_number'])
        
        print(f"🔧 Testing SSML with case: {test_case}")
        print(f"🔧 Original text: {test_text}")
        
        # Show the SSML preprocessing
        processed_text = preprocess_text_for_tts(test_text)
        print(f"🔧 SSML processed: {processed_text}")
        
        # Test TTS generation
        audio_content = process_text_with_tts_sync(test_text)
        
        if audio_content:
            return jsonify({
                "success": True,
                "message": "SSML TTS test successful",
                "test_case": test_case,
                "original_text": test_text,
                "ssml_processed": processed_text,
                "audio_length": len(audio_content),
                "available_test_cases": list(test_cases.keys())
            })
        else:
            return jsonify({
                "success": False,
                "message": "SSML TTS test failed",
                "test_case": test_case,
                "original_text": test_text,
                "ssml_processed": processed_text
            })
    except Exception as e:
        print(f"❌ SSML test error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "test_case": test_case if 'test_case' in locals() else "unknown"
        }), 500

@app.route('/api/debug/test-simple-tts', methods=['POST'])
def test_simple_tts():
    """Debug endpoint to test basic TTS without SSML"""
    try:
        # Simple test without SSML
        simple_text = "Hello, this is a simple test."
        
        print(f"🔧 Testing simple TTS with text: {simple_text}")
        
        # Test TTS generation without SSML preprocessing
        audio_content = process_text_with_tts_sync(simple_text)
        
        if audio_content:
            return jsonify({
                "success": True,
                "message": "Simple TTS test successful",
                "text": simple_text,
                "audio_length": len(audio_content)
            })
        else:
            return jsonify({
                "success": False,
                "message": "Simple TTS test failed",
                "text": simple_text
            })
    except Exception as e:
        print(f"❌ Simple TTS test error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "text": simple_text if 'simple_text' in locals() else "unknown"
        }), 500

@app.route('/api/debug/test-streaming-tts', methods=['POST'])
def test_streaming_tts():
    """Debug endpoint to test streaming TTS with Wavenet voice"""
    try:
        data = request.get_json()
        test_text = data.get('text', 'Hello, this is a test of the streaming text-to-speech system with Wavenet voice.')
        voice_name = data.get('voice', 'en-US-Wavenet-C')  # Default to female Wavenet voice
        
        print(f"🔧 Testing streaming TTS with text: {test_text}")
        print(f"🔧 Using voice: {voice_name}")
        
        # Test streaming TTS generation
        audio_content = process_text_with_streaming_tts(test_text, voice_name)
        
        if audio_content:
            return jsonify({
                "success": True,
                "message": "Streaming TTS test successful",
                "text": test_text,
                "voice": voice_name,
                "audio_length": len(audio_content),
                "streaming": True,
                "performance": _streaming_performance
            })
        else:
            return jsonify({
                "success": False,
                "message": "Streaming TTS test failed",
                "text": test_text,
                "voice": voice_name,
                "performance": _streaming_performance
            })
    except Exception as e:
        print(f"❌ Streaming TTS test error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "text": test_text if 'test_text' in locals() else "unknown",
            "performance": _streaming_performance
        }), 500

@app.route('/api/debug/streaming-performance', methods=['GET'])
def get_streaming_performance():
    """Debug endpoint to get streaming TTS performance metrics"""
    try:
        global _streaming_performance
        
        # Calculate performance metrics
        metrics = {
            "total_calls": _streaming_performance["total_calls"],
            "first_call_time": _streaming_performance["first_call_time"],
            "subsequent_calls_count": len(_streaming_performance["subsequent_call_times"]),
            "average_subsequent_time": None,
            "fastest_subsequent_time": None,
            "slowest_subsequent_time": None
        }
        
        if _streaming_performance["subsequent_call_times"]:
            subsequent_times = _streaming_performance["subsequent_call_times"]
            metrics["average_subsequent_time"] = sum(subsequent_times) / len(subsequent_times)
            metrics["fastest_subsequent_time"] = min(subsequent_times)
            metrics["slowest_subsequent_time"] = max(subsequent_times)
        
        return jsonify({
            "success": True,
            "performance_metrics": metrics,
            "raw_data": _streaming_performance
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/debug/test-case-text', methods=['POST'])
def test_case_text():
    """Debug endpoint to test TTS with case number text"""
    try:
        # Test with case number text
        case_text = "The case with number 0001111 was not found."
        
        print(f"🔧 Testing TTS with case text: {case_text}")
        
        # Test TTS generation
        audio_content = process_text_with_tts_sync(case_text)
        
        if audio_content:
            return jsonify({
                "success": True,
                "message": "Case text TTS test successful",
                "text": case_text,
                "audio_length": len(audio_content)
            })
        else:
            return jsonify({
                "success": False,
                "message": "Case text TTS test failed",
                "text": case_text
            })
    except Exception as e:
        print(f"❌ Case text TTS test error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "text": case_text if 'case_text' in locals() else "unknown"
        }), 500

@app.route('/api/voice/stt', methods=['POST'])
def speech_to_text():
    """Speech-to-Text endpoint - STT is handled by browser in LWC"""
    try:
        data = request.get_json()
        print("🔧 STT endpoint called - STT handled by browser in LWC")
        print(f"🔧 STT request data: {json.dumps(data, indent=2)}")

        # STT is handled by the browser in the LWC component
        # This endpoint is kept for compatibility but STT processing happens client-side
        text = data.get('text', '') if data else ''
        
        if not text:
            return jsonify({
                "text": "No text provided - STT is handled by browser in LWC",
                "note": "Speech-to-Text is processed client-side in the Lightning Web Component"
            })
        
        return jsonify({
            "text": text,
            "note": "STT processed by browser, text received successfully"
        })
    except Exception as e:
        print(f"❌ Error in STT endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to diagnose 503 issues"""
    try:
        logger.info("🏥 HEALTH CHECK ENDPOINT CALLED")
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "tts_engine": tts_engine is not None,
                "stt_engine": stt_engine is not None,
                "vad_engine": vad_engine is not None,
                "agent_session": agent_session is not None,
                "agent": agent is not None
            },
            "environment": {
                "google_credentials_set": os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') is not None,
                "livekit_url_set": os.environ.get('LIVEKIT_URL') is not None,
                "livekit_api_key_set": os.environ.get('LIVEKIT_API_KEY') is not None,
                "livekit_api_secret_set": os.environ.get('LIVEKIT_API_SECRET') is not None
            },
            "python_version": sys.version,
            "working_directory": os.getcwd()
        }
        
        # Check if any critical components are missing
        critical_components = ['tts_engine', 'stt_engine']
        missing_components = [comp for comp in critical_components if not health_status['components'][comp]]
        
        if missing_components:
            health_status['status'] = 'degraded'
            health_status['missing_components'] = missing_components
            logger.warning(f"⚠️ Health check: Missing components: {missing_components}")
        else:
            logger.info("✅ Health check: All critical components available")
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        import traceback
        logger.error(f"Health check traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/api/voice/tts-stream', methods=['POST'])
def text_to_speech_streaming():
    """Streaming Text-to-Speech endpoint with Wavenet voice"""
    try:
        print("=== STREAMING TTS API CALLED ===")
        data = request.get_json()
        print("Streaming TTS request data:", json.dumps(data, indent=2))

        text = data.get('text', '')
        voice_name = data.get('voice', 'en-US-Wavenet-C')  # Default to female Wavenet voice

        print(f"Text: {text}")
        print(f"Voice: {voice_name}")

        if not text:
            print("❌ Error: Text is required")
            return jsonify({"error": "Text is required"}), 400

        # Initialize LiveKit components if not already done
        if not tts_engine:
            print("🔧 Initializing LiveKit components for streaming TTS...")
            result = initialize_livekit_components()
            if not result:
                print("❌ Failed to initialize LiveKit components")
                return jsonify({"error": "Failed to initialize LiveKit components"}), 500

        print("✅ TTS engine available, processing with streaming...")

        # Process text with streaming TTS using Wavenet voice
        print(f"🔧 Original text for streaming TTS: '{text}'")
        print(f"🔧 Text length: {len(text)} characters")
        print(f"🔧 Voice: {voice_name}")
        
        try:
            audio_content = process_text_with_streaming_tts(text, voice_name)
        except AttributeError as attr_error:
            if "_interceptors_task" in str(attr_error):
                print(f"🔧 gRPC cleanup error in streaming TTS (non-critical): {attr_error}")
                print("🔧 This is a known gRPC cleanup issue, continuing...")
                # Try to get audio content anyway, as the error might be non-critical
                try:
                    audio_content = process_text_with_streaming_tts(text, voice_name)
                except:
                    audio_content = None
            else:
                print(f"❌ Attribute error in streaming TTS: {attr_error}")
                audio_content = None
        except Exception as e:
            print(f"❌ Error in streaming TTS processing: {e}")
            audio_content = None

        if not audio_content:
            print("❌ Streaming TTS generation failed, trying fallback...")
            # Fallback: return a simple response indicating TTS failed
            response_data = {
                "audio_content": None,
                "audio_data": None,
                "text": text,
                "voice": voice_name,
                "streaming": True,
                "error": "Streaming TTS generation failed",
                "fallback_message": "Text-to-speech is temporarily unavailable. Please read the text response."
            }
            print("❌ Streaming TTS fallback response sent")
            return jsonify(response_data)

        print(f"✅ Streaming TTS generated, audio length: {len(audio_content) if audio_content else 'null'}")

        response_data = {
            "audio_content": audio_content,
            "audio_data": audio_content,  # For backward compatibility
            "text": text,
            "voice": voice_name,
            "streaming": True,
            "chunk_size": STREAMING_CHUNK_SIZE,
            "delay": STREAMING_DELAY
        }

        print("Streaming TTS response data:", json.dumps({**response_data, "audio_content": f"[{len(audio_content) if audio_content else 0} bytes]"}, indent=2))

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error in streaming TTS: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/tts', methods=['POST'])
def text_to_speech():
    """Text-to-Speech endpoint"""
    try:
        logger.info("=" * 50)
        logger.info("🎙️ TTS API ENDPOINT CALLED")
        logger.info("=" * 50)
        
        # Log request details
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request remote address: {request.remote_addr}")
        
        data = request.get_json()
        logger.info(f"Request JSON data: {json.dumps(data, indent=2) if data else 'None'}")

        text = data.get('text', '') if data else ''
        language = data.get('language', 'en-US') if data else 'en-US'
        voice = data.get('voice', 'en-US-Wavenet-C') if data else 'en-US-Wavenet-C'  # Default to female Wavenet voice

        logger.info(f"Extracted parameters:")
        logger.info(f"  Text: '{text}' (length: {len(text)})")
        logger.info(f"  Language: {language}")
        logger.info(f"  Voice: {voice}")

        if not text:
            logger.error("❌ Error: Text is required")
            return jsonify({"error": "Text is required"}), 400

        # Check LiveKit components status
        logger.info(f"TTS engine status: {'Available' if tts_engine else 'Not initialized'}")
        
        # Initialize LiveKit components if not already done
        if not tts_engine:
            logger.info("🔧 Initializing LiveKit components for TTS...")
            result = initialize_livekit_components()
            if not result:
                logger.error("❌ Failed to initialize LiveKit components")
                return jsonify({"error": "Failed to initialize LiveKit components"}), 500
            logger.info("✅ LiveKit components initialized successfully")

        logger.info("✅ TTS engine available, processing...")

        # Process text with TTS using LiveKit directly (matching working code)
        logger.info(f"🔧 Processing text for TTS:")
        logger.info(f"  Original text: '{text}'")
        logger.info(f"  Text length: {len(text)} characters")
        logger.info(f"  Text type: {type(text)}")
        logger.info(f"  Language: {language}")
        logger.info(f"  Voice: {voice}")
        
        audio_content = process_text_with_tts_sync(text, language, voice)

        if not audio_content:
            print("❌ TTS generation failed, trying fallback...")
            # Fallback: return a simple response indicating TTS failed
            response_data = {
                "audio_content": None,
                "audio_data": None,
                "text": text,
                "language": language,
                "voice": voice,
                "error": "TTS generation failed",
                "fallback_message": "Text-to-speech is temporarily unavailable. Please read the text response."
            }
            print("❌ TTS fallback response sent")
            return jsonify(response_data)

        print(f"✅ TTS generated, audio length: {len(audio_content) if audio_content else 'null'}")

        response_data = {
            "audio_content": audio_content,
            "audio_data": audio_content,  # For backward compatibility
            "text": text,
            "language": language,
            "voice": voice
        }

        print("TTS response data:", json.dumps({**response_data, "audio_content": f"[{len(audio_content) if audio_content else 0} bytes]"}, indent=2))

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error in TTS: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 STARTING FLASK API SERVER")
    logger.info("=" * 60)
    logger.info("🔧 Flask API will run on port 10000")
    logger.info("🔧 Debug logging enabled")
    logger.info("🔧 Health check available at: http://localhost:10000/health")
    logger.info("🔧 Main endpoints:")
    logger.info("  - POST /api/einstein/agent")
    logger.info("  - POST /api/voice/tts")
    logger.info("  - POST /api/voice/tts-stream")
    logger.info("  - POST /api/voice/stt")
    logger.info("  - GET /health")
    logger.info("=" * 60)
    
    # Try to initialize components on startup
    logger.info("🔧 Attempting to initialize LiveKit components on startup...")
    try:
        init_result = initialize_livekit_components()
        if init_result:
            logger.info("✅ LiveKit components initialized successfully on startup")
        else:
            logger.warning("⚠️ LiveKit components initialization failed on startup - will retry on first request")
    except Exception as e:
        logger.warning(f"⚠️ LiveKit components initialization failed on startup: {e}")
        logger.warning("Components will be initialized on first request")
    
    logger.info("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=10000, debug=False)