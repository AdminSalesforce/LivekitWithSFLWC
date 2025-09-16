import os
import sys
import json
import base64
import tempfile
import subprocess
import asyncio
import logging
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except ImportError:
    print("Warning: flask-cors not installed, CORS may not work properly")
    CORS = None
from livekit.plugins import google
from livekit.agents import Agent, AgentSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
llm_engine = None

# TTS engine cache to store TTS engines for different voices
tts_engine_cache = {}

def process_text_with_tts_sync(text, language='en-US', voice='en-US-Wavenet-A'):
    """Synchronous wrapper for TTS processing using subprocess"""
    try:
        print("🔧 process_text_with_tts_sync FUNCTION CALLED")
        print(f"🔧 Text: {text[:50]}...")
        print(f"🔧 Language: {language}, Voice: {voice}")
        
        # Create temporary file with text
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(text)
            temp_file_path = temp_file.name

        try:
            # Create a simple TTS script
            tts_script = f"""
import os
import sys
import json
import base64
import tempfile
import asyncio
from livekit.plugins import google

# Set up Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "{os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}"

async def generate_tts():
   try:
       # Initialize TTS engine
       tts_engine = google.TTS()
       
       # Read text from file
       with open('{temp_file_path}', 'r') as f:
           text = f.read()
       
       print(f"Processing text: {{text[:50]}}...")
       
       # Generate audio
       audio_stream = tts_engine.synthesize(text=text)
       
       audio_chunks = []
       try:
           # Try async iteration first
           async for chunk in audio_stream:
               if hasattr(chunk, 'frame') and chunk.frame:
                   audio_data = chunk.frame.data
                   audio_chunks.append(audio_data)
                   print(f"Collected chunk: {{len(audio_data)}} bytes")
       except:
           # Fallback to sync iteration
           for chunk in audio_stream:
               if hasattr(chunk, 'frame') and chunk.frame:
                   audio_data = chunk.frame.data
                   audio_chunks.append(audio_data)
                   print(f"Collected chunk: {{len(audio_data)}} bytes")
       
       if not audio_chunks:
           print("No audio chunks collected")
           return None
       
       # Combine audio data
       full_audio_bytes = b"".join(audio_chunks)
       print(f"Total audio bytes: {{len(full_audio_bytes)}}")
       
       # Create WAV file
       wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
           b'RIFF', 36 + len(full_audio_bytes), b'WAVE', b'fmt ', 16, 1,
           1, 24000, 24000 * 1 * 16 // 8, 1 * 16 // 8, 16, b'data', len(full_audio_bytes)
       )
       wav_audio = wav_header + full_audio_bytes
       
       # Convert to base64
       audio_base64 = base64.b64encode(wav_audio).decode('utf-8')
       print(f"WAV audio created: {{len(wav_audio)}} bytes")
       
       return audio_base64
       
   except Exception as e:
       print(f"Error in TTS: {{e}}")
       import traceback
       traceback.print_exc()
       return None

if __name__ == "__main__":
   import struct
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
                # Run the script
                print("🔧 Running TTS subprocess...")
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
                    print("✅ TTS subprocess completed successfully")
                    print(f"✅ Audio data length: {len(audio_base64)} characters")
                    return audio_base64
                else:
                    print("❌ TTS subprocess failed")
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

async def process_text_with_tts_async(text, language='en-US', voice='en-US-Wavenet-A'):  # pylint: disable=unused-argument
    """Async TTS processing using LiveKit directly"""
    try:
        print(f"🔧 Async TTS processing for text: {text[:50]}...")

        if not tts_engine:
            print("❌ TTS engine not available in async function")
            return None

        # Process text using LiveKit TTS directly
        audio_stream = tts_engine.synthesize(text=text)

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
                for chunk in audio_stream:
                    if hasattr(chunk, 'frame') and chunk.frame:
                        audio_data = chunk.frame.data
                        audio_chunks.append(audio_data)
                        print(f"🔧 Collected audio chunk: {len(audio_data)} bytes")
                    elif hasattr(chunk, 'data'):
                        audio_data = chunk.data
                        audio_chunks.append(audio_data)
                        print(f"🔧 Collected audio chunk (data): {len(audio_data)} bytes")
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

def initialize_livekit_components():
    """Initialize LiveKit components"""
    global stt_engine, tts_engine, vad_engine, agent_session, agent
    
    try:
        print("🔧 Initializing LiveKit components...")
        print(f"🔧 TTS engine available: {tts_engine is not None}")
        print(f"🔧 Agent session available: {agent_session is not None}")
        
        # Setup Google credentials
        if not setup_google_credentials():
            print("❌ Failed to setup Google credentials")
            return False
        
        # Initialize STT engine
        try:
            print("🔧 Initializing STT engine...")
            stt_engine = google.STT()
            print("✅ STT engine initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize STT engine: {e}")
            return False
        
        # Initialize TTS engine with default voice
        try:
            print("🔧 Initializing TTS engine...")
            tts_engine = google.TTS()
            print("✅ TTS engine initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize TTS engine: {e}")
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
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "LiveKit Voice Agent API is running",
        "components": {
            "stt_engine": stt_engine is not None,
            "tts_engine": tts_engine is not None,
            "agent_session": agent_session is not None,
            "agent": agent is not None
        }
    })

@app.route('/api/einstein/agent', methods=['POST'])
def einstein_agent():
    """Salesforce Einstein Agent endpoint"""
    try:
        print("=== EINSTEIN AGENT API CALLED ===")
        data = request.get_json()
        print("Einstein request data:", json.dumps(data, indent=2))
        
        message = data.get('message', '')
        session_id = data.get('session_id', '')
        agent_id = data.get('agent_id', 'agent_001')
        
        print(f"Message: {message}")
        print(f"Session ID: {session_id}")
        print(f"Agent ID: {agent_id}")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Simulate Salesforce Einstein Agent response
        # In a real implementation, this would call the actual Salesforce Einstein API
        response_message = f"I'm sorry, I'm having trouble connecting to Salesforce right now. Your message was: '{message}'. Please try again later."
        
        return jsonify({
            "message": response_message,
            "session_id": session_id,
            "agent_id": agent_id,
            "success": True
        })
        
    except Exception as e:
        print(f"❌ Error in Einstein Agent: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route('/api/voice/stt', methods=['POST'])
def speech_to_text():
    """Speech-to-Text endpoint"""
    try:
        data = request.get_json()
        # audio_data = data.get('audio_data')  # Will be used in actual STT implementation

        # Initialize LiveKit components if not already done
        if not stt_engine:
            result = initialize_livekit_components()
            if not result:
                return jsonify({"error": "Failed to initialize LiveKit components"}), 500

        # Process audio with STT
        # Note: This is a simplified example - actual implementation would process audio data
        text = "Speech converted to text"  # Placeholder for actual STT processing

        return jsonify({
            "text": text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/tts', methods=['POST'])
def text_to_speech():
    """Text-to-Speech endpoint"""
    try:
        print("=== TTS API CALLED ===")
        data = request.get_json()
        print("TTS request data:", json.dumps(data, indent=2))

        text = data.get('text', '')
        language = data.get('language', 'en-US')
        voice = data.get('voice', 'en-US-Wavenet-A')

        print(f"Text: {text}")
        print(f"Language: {language}")
        print(f"Voice: {voice}")

        if not text:
            print("❌ Error: Text is required")
            return jsonify({"error": "Text is required"}), 400

        # Initialize LiveKit components if not already done
        if not tts_engine:
            print("🔧 Initializing LiveKit components for TTS...")
            result = initialize_livekit_components()
            if not result:
                print("❌ Failed to initialize LiveKit components")
                return jsonify({"error": "Failed to initialize LiveKit components"}), 500

        print("✅ TTS engine available, processing...")

        # Process text with TTS using synchronous wrapper
        audio_content = process_text_with_tts_sync(text, language, voice)

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
    print("🔧 Starting Flask API server...")
    print("🔧 Flask API will run on port 10000")
    app.run(host='0.0.0.0', port=10000, debug=False)