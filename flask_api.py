#!/usr/bin/env python3
"""
Flask API wrapper for the Salesforce Voice Agent with Einstein Agent and LiveKit
"""

from flask import Flask, request, jsonify
import os
import aiohttp
import asyncio
import json
import uuid
import tempfile
import logging
import time
import base64
from flask_cors import CORS

# Initialize logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle Google credentials for cloud deployment BEFORE LiveKit imports
def setup_google_credentials():
    """Setup Google credentials for both local and cloud deployment"""
    print("🔧 Starting Google credentials setup...")
    try:
        # Method 1: Try Render secret file (for cloud deployment)
        secret_file_path = "/etc/secrets/google-credentials.json"
        print(f"🔧 Checking secret file: {secret_file_path}")
        if os.path.exists(secret_file_path):
            print(f"✅ Secret file found: {secret_file_path}")
            
            # Fix the private key format in the credentials file
            try:
                with open(secret_file_path, 'r') as f:
                    creds_data = json.load(f)
                
                # Fix private key format - ensure it has proper line breaks
                if 'private_key' in creds_data:
                    private_key = creds_data['private_key']
                    # Replace escaped newlines with actual newlines
                    if '\\n' in private_key:
                        private_key = private_key.replace('\\n', '\n')
                        # Ensure proper PEM format
                        if not private_key.startswith('-----BEGIN'):
                            private_key = '-----BEGIN PRIVATE KEY-----\n' + private_key + '\n-----END PRIVATE KEY-----'
                        creds_data['private_key'] = private_key
                        
                    # Write the fixed credentials to a temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_f:
                        json.dump(creds_data, temp_f)
                        temp_creds_path = temp_f.name
                        
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_creds_path
                    print(f"✅ Fixed private key format and set credentials: {temp_creds_path}")
                    print(f"✅ Environment variable set to: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
                    logger.info(f"✅ Google credentials set from Render secret file with fixed private key: {temp_creds_path}")
                    return True
            except Exception as e:
                print(f"❌ Error fixing credentials file: {e}")
                # Fallback to original file
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = secret_file_path
                logger.info(f"✅ Google credentials set from Render secret file: {secret_file_path}")
                return True
        else:
            print(f"❌ Secret file not found: {secret_file_path}")
        
        # Method 2: Try local google-credentials.json file
        print("🔧 Checking local file: google-credentials.json")
        if os.path.exists("google-credentials.json"):
            print("✅ Local file found: google-credentials.json")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-credentials.json"
            logger.info("✅ Google credentials set from local file: google-credentials.json")
            return True
        else:
            print("❌ Local file not found: google-credentials.json")
        
        # Method 3: Try JSON string from environment variable (fallback)
        if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
            creds_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
            logger.info("Found GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")
            try:
                creds_data = json.loads(creds_json)
                
                # Create temporary file for Google credentials
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(creds_data, f)
                    creds_path = f.name
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                    logger.info(f"✅ Google credentials set from JSON string: {creds_path}")
                    return True
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
                return False
        
        # Method 4: Try direct file path (for local deployment)
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google-credentials.json")
        if os.path.exists(creds_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            logger.info(f"✅ Google credentials set from file: {creds_path}")
            return True
        
        # No credentials found
        print("❌ No Google credentials found!")
        logger.error("❌ No Google credentials found!")
        logger.error("Please add google-credentials.json as a secret file in Render")
        logger.error("Or set GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")
        return False
        
    except Exception as e:
        print(f"❌ Failed to setup Google credentials: {e}")
        logger.error(f"❌ Failed to setup Google credentials: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize Google credentials BEFORE LiveKit imports
print("🔧 Setting up Google credentials...")
google_creds_ok = setup_google_credentials()
print(f"🔧 Google credentials setup result: {google_creds_ok}")

# Set Google API key for TTS
if google_creds_ok:
    try:
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
        print(f"✅ Google API key set: {os.environ['GOOGLE_API_KEY'][:10]}...")
    except Exception as e:
        print(f"❌ Failed to set Google API key: {e}")

# Set LiveKit environment variables
os.environ["LIVEKIT_API_KEY"] = os.getenv("LIVEKIT_API_KEY", "YOUR_LIVEKIT_API_KEY")
os.environ["LIVEKIT_API_SECRET"] = os.getenv("LIVEKIT_API_SECRET", "YOUR_LIVEKIT_API_SECRET")
os.environ["LIVEKIT_URL"] = os.getenv("LIVEKIT_URL", "YOUR_LIVEKIT_URL")

# Verify credentials are properly set before LiveKit imports
print(f"🔧 Final GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
print(f"🔧 File exists: {os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''))}")

# LiveKit imports - must be at module level for plugin registration
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, Agent, ConversationItemAddedEvent, UserInputTranscribedEvent
from livekit.agents.voice import AgentSession
from livekit.plugins import google
from livekit import rtc

app = Flask(__name__)
CORS(app)

# Enable LiveKit debug logs
os.environ["LIVEKIT_LOG_LEVEL"] = "debug"
logging.getLogger("livekit").setLevel(logging.DEBUG)

# Salesforce configuration
SALESFORCE_DOMAIN = os.getenv("SALESFORCE_ORG_DOMAIN", "https://de1740385138027.my.salesforce.com")
SALESFORCE_CLIENT_ID = os.getenv("SALESFORCE_CLIENT_ID", "YOUR_SALESFORCE_CLIENT_ID")
SALESFORCE_CLIENT_SECRET = os.getenv("SALESFORCE_CLIENT_SECRET", "YOUR_SALESFORCE_CLIENT_SECRET")

# LiveKit components (will be initialized when needed)
stt_engine = None
tts_engine = None
vad_engine = None
agent_session = None
agent = None
llm_engine = None

def initialize_livekit_components():
    """Initialize LiveKit components for voice processing"""
    global stt_engine, tts_engine, vad_engine, agent_session, agent, llm_engine
    
    print("🚀 Starting LiveKit components initialization...")
    
    try:
        # Check if Google credentials are properly set
        if not google_creds_ok:
            print("❌ Google credentials not available, skipping LiveKit initialization")
            logger.warning("Google credentials not available, skipping LiveKit initialization")
            return False
        
        # Verify Google credentials file exists and is readable
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            print(f"❌ Google credentials file not found: {creds_path}")
            logger.error(f"Google credentials file not found: {creds_path}")
            return False
        
        # Debug: Check Google credentials before initializing
        print(f"Before LiveKit init - GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        print(f"Before LiveKit init - File exists: {os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''))}")
        
        # Try to read the credentials file to verify it's valid JSON
        try:
            with open(creds_path, 'r') as f:
                creds_content = f.read()
                print(f"Credentials file size: {len(creds_content)} characters")
                # Try to parse as JSON to verify it's valid
                json.loads(creds_content)
                print("✅ Credentials file is valid JSON")
        except Exception as e:
            print(f"❌ Error reading credentials file: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Initialize Google STT (let it use GOOGLE_APPLICATION_CREDENTIALS automatically)
        try:
            print("🔧 Initializing Google STT...")
            stt_engine = google.STT(
                model="latest_long",
                spoken_punctuation=True,
                languages="en-US",
            )
            print("✅ Google STT initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Google STT: {e}")
            raise e
        
        # Initialize Google TTS (let it use GOOGLE_APPLICATION_CREDENTIALS automatically)
        try:
            print("🔧 Initializing Google TTS...")
            tts_engine = google.TTS()
            print("✅ Google TTS initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Google TTS: {e}")
            raise e
        
        # VAD (Voice Activity Detection) is not available in current LiveKit version
        # Skip VAD initialization
        print("🔧 Skipping VAD initialization (not available in current LiveKit version)")
        vad_engine = None
        
        # Create AgentSession with STT and TTS
        try:
            print("🔧 Creating AgentSession...")
            # Create a new event loop for AgentSession
            import asyncio
            import threading
            
            def create_agent_session():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return AgentSession(
                    stt=stt_engine,
                    tts=tts_engine,
                )
            
            # Run AgentSession creation in a separate thread with its own event loop
            agent_session = create_agent_session()
            print("✅ AgentSession created successfully")
        except Exception as e:
            print(f"❌ Failed to create AgentSession: {e}")
            raise e
        
        # Create Agent
        try:
            print("🔧 Creating Agent...")
            agent = Agent(
                instructions="You are a helpful Salesforce voice assistant. Help users with their Salesforce cases and questions. Always provide helpful and engaging responses."
            )
            print("✅ Agent created successfully")
        except Exception as e:
            print(f"❌ Failed to create Agent: {e}")
            raise e
        
        logger.info("LiveKit components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LiveKit components: {e}")
        import traceback
        traceback.print_exc()
        return False

# SalesforceLLM class will be defined dynamically in initialize_livekit_components()
# to avoid import-time dependencies
SalesforceLLM = None

async def call_einstein_agent(message, session_id, agent_id=None):
    """Call Salesforce Einstein Agent API with dynamic agent ID"""
    try:
        # Agent ID is required - must be provided dynamically
        if not agent_id:
            return "Error: Agent ID is required but not provided"
        
        # Get Salesforce access token
        access_token = await get_salesforce_access_token()
        if not access_token:
            return "I'm sorry, I'm having trouble connecting to Salesforce right now."
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Use the provided agent_id dynamically
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/agents/{agent_id}/sessions/{session_id}/messages"
        
        payload = {
            "message": {
                "sequenceId": str(int(time.time() * 1000)),
                "type": "Text",
                "text": message
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                logger.info(f"Salesforce API response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Salesforce response: {result}")
                    
                    if 'message' in result:
                        return result.get('message', 'No response message')
                    elif 'messages' in result:
                        messages = result.get('messages', [])
                        if messages and len(messages) > 0:
                            message_obj = messages[0]
                            message_type = message_obj.get('type', '')
                            
                            if message_type == 'Inform':
                                return message_obj.get('text', 'No response text')
                            else:
                                return str(message_obj)
                        else:
                            return 'No response messages'
                    else:
                        return result.get('text', 'No response text')
                else:
                    text = await response.text()
                    logger.error(f"Salesforce call failed: {text}")
                    return "I'm sorry, I'm having trouble connecting to Salesforce right now."
    except Exception as e:
        print(f"Error calling Einstein Agent: {e}")
        return "I'm sorry, I encountered an error processing your request."

async def get_salesforce_access_token():
    """Get Salesforce access token using OAuth2 Client Credentials flow"""
    try:
        auth_url = f"{SALESFORCE_DOMAIN}/services/oauth2/token"
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': SALESFORCE_CLIENT_ID,
            'client_secret': SALESFORCE_CLIENT_SECRET
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    result = await response.json()
                    access_token = result["access_token"]
                    print("Successfully obtained Salesforce access token")
                    return access_token
                else:
                    error_text = await response.text()
                    print(f"Failed to get access token: {response.status} - {error_text}")
                    raise Exception(f"Failed to get Salesforce access token: {response.status}")
    except Exception as e:
        print(f"Error getting Salesforce access token: {e}")
        return None

@app.route('/')
def health_check():
    """Health check endpoint"""
    # Don't automatically initialize LiveKit components in health check
    # This prevents the error from occurring on every request
    
    return jsonify({
        "status": "running",
        "message": "Salesforce Voice Agent API with Einstein Agent and LiveKit is running",
        "google_credentials": {
            "available": google_creds_ok,
            "path": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "Not set"),
            "local_file_exists": os.path.exists("google-credentials.json"),
            "secret_file_exists": os.path.exists("/etc/secrets/google-credentials.json"),
            "json_env_var": "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ
        },
        "livekit_components": {
            "stt": stt_engine is not None,
            "tts": tts_engine is not None,
            "vad": vad_engine is not None,
            "agent_session": agent_session is not None,
            "agent": agent is not None,
            "llm_engine": llm_engine is not None
        },
        "salesforce_domain": SALESFORCE_DOMAIN,
        "agent_id": "dynamic (provided in requests)",
        "livekit_url": os.environ.get("LIVEKIT_URL", "Not set"),
        "endpoints": {
            "health": "/",
            "chat": "/api/chat",
            "einstein_agent": "/api/einstein/agent",
            "voice_process": "/api/voice/process",
            "voice_stt": "/api/voice/stt",
            "voice_tts": "/api/voice/tts"
        }
    })

@app.route('/health')
def health():
    """Health endpoint for Render"""
    return jsonify({"status": "healthy"})

@app.route('/api/debug/credentials')
def debug_credentials():
    """Debug endpoint to check Google credentials setup"""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "Not set")
    
    return jsonify({
        "google_application_credentials": creds_path,
        "file_exists": os.path.exists(creds_path) if creds_path != "Not set" else False,
        "secret_file_exists": os.path.exists("/etc/secrets/google-credentials.json"),
        "local_file_exists": os.path.exists("google-credentials.json"),
        "json_env_var_exists": "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ,
        "setup_result": google_creds_ok
    })

@app.route('/api/debug/init-livekit')
def debug_init_livekit():
    """Debug endpoint to manually initialize LiveKit components"""
    try:
        print("🔧 Manual LiveKit initialization requested...")
        print("🔧 Current Google credentials status:")
        print(f"  - GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
        print(f"  - File exists: {os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', ''))}")
        print(f"  - Google creds OK: {google_creds_ok}")
        
        result = initialize_livekit_components()
        print(f"🔧 Manual LiveKit initialization result: {result}")
        
        return jsonify({
            "success": result,
            "stt_engine": stt_engine is not None,
            "tts_engine": tts_engine is not None,
            "vad_engine": vad_engine is not None,
            "agent_session": agent_session is not None,
            "agent": agent is not None,
            "llm_engine": llm_engine is not None,
            "google_credentials_path": os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'),
            "google_credentials_file_exists": os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')),
            "google_creds_ok": google_creds_ok
        })
    except Exception as e:
        print(f"❌ Manual LiveKit initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "google_credentials_path": os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'),
            "google_credentials_file_exists": os.path.exists(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')),
            "google_creds_ok": google_creds_ok
        })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for text-based conversations"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        agent_id = data.get('agent_id')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not agent_id:
            return jsonify({"error": "Agent ID is required"}), 400
        
        # Call Einstein Agent
        result = asyncio.run(call_einstein_agent(message, session_id, agent_id))
        
        return jsonify({
            "response": result,
            "session_id": session_id,
            "agent_id": agent_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/einstein/agent', methods=['POST'])
def einstein_agent():
    """Einstein Agent endpoint"""
    try:
        print("=== EINSTEIN AGENT API CALLED ===")
        data = request.get_json()
        print("Request data:", json.dumps(data, indent=2))
        
        message = data.get('message', '')
        agent_id = data.get('agent_id')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        print(f"Message: {message}")
        print(f"Session ID: {session_id}")
        print(f"Agent ID: {agent_id}")
        
        if not agent_id:
            print("❌ Error: Agent ID is required")
            return jsonify({"error": "Agent ID is required"}), 400
        
        print(f"✅ Calling Einstein Agent with ID: {agent_id}")
        
        # Call Einstein Agent
        result = asyncio.run(call_einstein_agent(message, session_id, agent_id))
        
        print(f"✅ Einstein Agent response: {result}")
        
        response_data = {
            "response": result,
            "message": result,  # Add message field for LWC compatibility
            "session_id": session_id,
            "agent_id": agent_id
        }
        
        print("Response data:", json.dumps(response_data, indent=2))
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in Einstein Agent API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/process', methods=['POST'])
def process_voice():
    """Process voice input with STT, Einstein Agent, and TTS"""
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        agent_id = data.get('agent_id')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not agent_id:
            return jsonify({"error": "Agent ID is required"}), 400
        
        # Initialize LiveKit components if not already done
        if not stt_engine or not tts_engine:
            result = initialize_livekit_components()
            if not result:
                return jsonify({"error": "Failed to initialize LiveKit components"}), 500
        
        # Process voice with STT
        # Note: This is a simplified example - actual implementation would process audio data
        text = "Voice input processed"  # Placeholder for actual STT processing
        
        # Call Einstein Agent
        response = asyncio.run(call_einstein_agent(text, session_id, agent_id))
        
        # Process response with TTS
        # Note: This is a simplified example - actual implementation would generate audio
        audio_response = "Audio response generated"  # Placeholder for actual TTS processing
        
        return jsonify({
            "text": text,
            "response": response,
            "audio_response": audio_response,
            "session_id": session_id,
            "agent_id": agent_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def process_text_with_tts(text, language='en-US', voice='en-US-Wavenet-A'):
    """Converts text to base64 audio data using Google TTS via LiveKit"""
    try:
        print(f"🔧 Processing TTS for text: {text[:50]}...")
        print(f"Language: {language}, Voice: {voice}")
        
        if not tts_engine:
            print("❌ TTS engine not available")
            return None
        
        # Use LiveKit TTS with specific voice settings
        audio_stream = tts_engine.synthesize(
            text=text,
            voice=voice,
            language=language
        )
        
        print("🔧 TTS synthesis started...")
        audio_chunks = []
        async for chunk in audio_stream:
            audio_chunks.append(chunk.data)
            print(f"🔧 Received audio chunk: {len(chunk.data)} bytes")
        
        full_audio_bytes = b"".join(audio_chunks)
        print(f"✅ TTS synthesis complete: {len(full_audio_bytes)} total bytes")
        
        # Encode to base64 for transmission
        audio_base64 = base64.b64encode(full_audio_bytes).decode('utf-8')
        print(f"✅ Audio encoded to base64: {len(audio_base64)} characters")
        
        return audio_base64
        
    except Exception as e:
        print(f"❌ Error in process_text_with_tts: {e}")
        logger.error(f"Error in process_text_with_tts: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/voice/stt', methods=['POST'])
def speech_to_text():
    """Speech-to-Text endpoint"""
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        
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
        
        # Process text with TTS
        audio_content = asyncio.run(process_text_with_tts(text, language, voice))
        
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)