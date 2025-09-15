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
from flask_cors import CORS

# LiveKit imports - moved to function level to prevent import-time initialization
# from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, Agent, ConversationItemAddedEvent, UserInputTranscribedEvent
# from livekit.agents.voice import AgentSession
# from livekit.plugins import google
# from livekit import rtc

app = Flask(__name__)
CORS(app)

# Enable LiveKit debug logs
os.environ["LIVEKIT_LOG_LEVEL"] = "debug"
logging.getLogger("livekit").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Set LiveKit environment variables
os.environ["LIVEKIT_API_KEY"] = os.getenv("LIVEKIT_API_KEY", "YOUR_LIVEKIT_API_KEY")
os.environ["LIVEKIT_API_SECRET"] = os.getenv("LIVEKIT_API_SECRET", "YOUR_LIVEKIT_API_SECRET")
os.environ["LIVEKIT_URL"] = os.getenv("LIVEKIT_URL", "YOUR_LIVEKIT_URL")

# Handle Google credentials for cloud deployment
def setup_google_credentials():
    """Setup Google credentials for both local and cloud deployment"""
    print("🔧 Starting Google credentials setup...")
    try:
        # Method 1: Try Render secret file (for cloud deployment)
        secret_file_path = "/etc/secrets/google-credentials.json"
        print(f"🔧 Checking secret file: {secret_file_path}")
        if os.path.exists(secret_file_path):
            print(f"✅ Secret file found: {secret_file_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = secret_file_path
            logger.info(f"✅ Google credentials set from Render secret file: {secret_file_path}")
            
            # Debug: Verify the file path and existence
            print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
            print(f"File exists: {os.path.exists(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])}")
            
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

# Setup Google credentials
print("🔧 Setting up Google credentials...")
google_creds_ok = setup_google_credentials()
print(f"🔧 Google credentials setup result: {google_creds_ok}")

# Set Google API key only if it's provided
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key and google_api_key != "YOUR_GOOGLE_API_KEY":
    os.environ["GOOGLE_API_KEY"] = google_api_key
    print(f"✅ Google API key set: {google_api_key[:10]}...")
else:
    print("⚠️ Google API key not set - this might cause issues with some Google services")

# Salesforce OAuth2 Connected App credentials
SALESFORCE_DOMAIN = os.environ.get('SALESFORCE_ORG_DOMAIN', 'YOUR_SALESFORCE_ORG_DOMAIN')
SALESFORCE_CLIENT_ID = os.environ.get('SALESFORCE_CLIENT_ID', 'YOUR_SALESFORCE_CLIENT_ID')
SALESFORCE_CLIENT_SECRET = os.environ.get('SALESFORCE_CLIENT_SECRET', 'YOUR_SALESFORCE_CLIENT_SECRET')
SALESFORCE_AGENT_ID = os.environ.get('SALESFORCE_AGENT_ID', None)  # No default - must be provided dynamically

# Global access token
access_token = None

# LiveKit Voice Agent components
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
        # Import LiveKit modules here to prevent import-time initialization errors
        print("🔧 Importing LiveKit modules...")
        from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, Agent, ConversationItemAddedEvent, UserInputTranscribedEvent
        from livekit.agents.voice import AgentSession
        from livekit.plugins import google
        from livekit import rtc
        print("✅ LiveKit modules imported successfully")
        
        # Now we can use the imported modules
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
        
        # Initialize Google VAD (Voice Activity Detection)
        try:
            print("🔧 Initializing Google VAD...")
            vad_engine = google.VAD()
            print("✅ Google VAD initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Google VAD: {e}")
            raise e
        
        # Initialize Salesforce LLM
        try:
            print("🔧 Initializing Salesforce LLM...")
            llm_engine = SalesforceLLM()
            print("✅ Salesforce LLM initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Salesforce LLM: {e}")
            raise e
        
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
        return False

class SalesforceLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self.last_message = ""
        self.access_token = None
        self.session_id = str(uuid.uuid4())
        self.salesforce_session_id = None
        self._session_started = False
    
    def set_user_message(self, message: str):
        self.last_message = message
        logger.info(f"User message: {message}")
    
    async def _authenticate_salesforce(self):
        if self.access_token:
            return
        
        auth_url = f"{SALESFORCE_DOMAIN}/services/oauth2/token"
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': SALESFORCE_CLIENT_ID,
            'client_secret': SALESFORCE_CLIENT_SECRET
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=auth_data) as response:
                if response.status == 200:
                    auth_result = await response.json()
                    self.access_token = auth_result['access_token']
                    logger.info("Salesforce authenticated")
                else:
                    logger.error(f"Salesforce auth failed: {response.status}")
    
    async def _start_salesforce_session(self):
        """Start a Salesforce Einstein Agent session"""
        await self._authenticate_salesforce()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/agents/{SALESFORCE_AGENT_ID}/sessions"
        payload = {
            "externalSessionKey": self.session_id,
            "instanceConfig": {
                "endpoint": SALESFORCE_DOMAIN
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
                "chunkTypes": ["Text"]
            },
            "bypassUser": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                logger.info(f"Start session response status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Session started successfully: {result}")
                    self.salesforce_session_id = result.get('sessionId')
                    logger.info(f"Stored Salesforce session ID: {self.salesforce_session_id}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Failed to start session: {text}")
                    return False

    async def _call_salesforce(self, message: str):
        await self._authenticate_salesforce()
        
        # Start session if not already started
        if not self._session_started:
            session_started = await self._start_salesforce_session()
            if not session_started:
                logger.warning("Failed to start Salesforce session, using fallback")
                return self._generate_fallback_response(message)
            self._session_started = True
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        salesforce_session_id = getattr(self, 'salesforce_session_id', self.session_id)
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/sessions/{salesforce_session_id}/messages"
        
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
                                return message_obj.get('message', 'No response message')
                            elif message_type == 'Failure':
                                return self._generate_fallback_response(message)
                            elif message_type == 'Text':
                                return message_obj.get('text', 'No response text')
                            else:
                                return message_obj.get('message', f'Received response type: {message_type}')
                        else:
                            return "No messages in response"
                    else:
                        return "No response message found"
                else:
                    logger.error(f"Salesforce API error: {response.status}")
                    return self._generate_fallback_response(message)
    
    def _generate_fallback_response(self, message: str):
        """Generate a fallback response when Salesforce API is unavailable"""
        return "I'm sorry, I'm having trouble connecting to Salesforce right now. Please try again in a moment."
    
    def chat(self, chat_ctx=None, messages=None, tools=None, tool_choice=None, conn_options=None, **kwargs):
        logger.info("Chat called")
        
        # Extract user message from different sources
        user_message = "No message received"
        
        if self.last_message:
            user_message = self.last_message
        elif chat_ctx and hasattr(chat_ctx, 'messages') and chat_ctx.messages:
            for msg in chat_ctx.messages:
                if hasattr(msg, 'role') and msg.role == "user":
                    if hasattr(msg, 'content'):
                        if isinstance(msg.content, list):
                            user_message = " ".join(msg.content)
                        else:
                            user_message = msg.content
                        break
                elif hasattr(msg, 'type') and msg.type == "transcript":
                    if hasattr(msg, 'text'):
                        user_message = msg.text
                        break
        
        logger.info(f"Final user message: {user_message}")
        return _ChatContext(self, user_message)

class _ChatContext(llm.LLMStream):
    def __init__(self, llm_instance: SalesforceLLM, user_message: str):
        self.llm_instance = llm_instance
        self.user_message = user_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _run(self):
        """Required abstract method implementation"""
        response = await self.llm_instance._call_salesforce(self.user_message)
        yield llm.ChatChunk(
            id=str(uuid.uuid4()),
            choices=[llm.ChoiceDelta(
                content=response,
                role="assistant"
            )]
        )

    async def __aiter__(self):
        async for chunk in self._run():
            yield chunk

async def process_voice_with_stt(audio_data):
    """Process audio data using Google STT"""
    try:
        if not stt_engine:
            initialize_livekit_components()
        
        # Convert audio data to text using Google STT
        # Note: This is a simplified example - actual implementation would depend on audio format
        transcript = await stt_engine.transcribe(audio_data)
        return transcript
    except Exception as e:
        logger.error(f"STT processing error: {e}")
        return None

async def process_text_with_tts(text):
    """Convert text to speech using Google TTS"""
    try:
        if not tts_engine:
            initialize_livekit_components()
        
        # Convert text to audio using Google TTS
        # Note: This is a simplified example - actual implementation would return audio data
        audio_data = await tts_engine.synthesize(text)
        return audio_data
    except Exception as e:
        logger.error(f"TTS processing error: {e}")
        return None

async def get_salesforce_access_token():
    """Get Salesforce access token using OAuth2 client credentials flow"""
    global access_token
    
    if access_token:
        return access_token

    token_url = f"{SALESFORCE_DOMAIN}/services/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": SALESFORCE_CLIENT_ID,
        "client_secret": SALESFORCE_CLIENT_SECRET,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(token_url, data=data) as response:
            if response.status == 200:
                result = await response.json()
                access_token = result["access_token"]
                print("Successfully obtained Salesforce access token")
                return access_token
            else:
                error_text = await response.text()
                print(f"Failed to get access token: {response.status} - {error_text}")
                raise Exception(f"Failed to get Salesforce access token: {response.status}")

async def call_einstein_agent(message, session_id, agent_id=None):
    """Call Salesforce Einstein Agent API with dynamic agent ID"""
    try:
        # Agent ID is required - must be provided dynamically
        if not agent_id:
            if SALESFORCE_AGENT_ID:
                agent_id = SALESFORCE_AGENT_ID
            else:
                return "Error: Agent ID is required but not provided"
        
        token = await get_salesforce_access_token()
        
        # Call Salesforce Einstein Agent with dynamic agent ID
        agent_url = f"{SALESFORCE_DOMAIN}/services/data/v58.0/einstein/agent/{agent_id}/chat"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": message,
            "sessionId": session_id
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(agent_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "I'm sorry, I couldn't process your request.")
                else:
                    error_text = await response.text()
                    print(f"Einstein Agent API error: {response.status} - {error_text}")
                    return "I'm sorry, I'm having trouble connecting to Salesforce right now."
    except Exception as e:
        print(f"Error calling Einstein Agent: {e}")
        return "I'm sorry, I encountered an error processing your request."

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
            "secret_file_exists": os.path.exists("/etc/secrets/google-credentials.json"),
            "local_file_exists": os.path.exists("google-credentials.json"),
            "json_env_var": "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ
        },
        "salesforce_domain": SALESFORCE_DOMAIN,
        "agent_id": "dynamic (provided in requests)",
        "livekit_url": os.environ.get("LIVEKIT_URL"),
        "livekit_components": {
            "stt": stt_engine is not None,
            "tts": tts_engine is not None,
            "vad": vad_engine is not None
        },
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

@app.route('/api/test', methods=['POST'])
def test_endpoint():
    """Test endpoint that doesn't require LiveKit"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        agent_id = data.get('agent_id')
        
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        # Test Salesforce connection without LiveKit
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(call_einstein_agent(message, "test-session", agent_id))
        loop.close()
        
        return jsonify({
            'response': response,
            'agent_id': agent_id,
            'livekit_available': stt_engine is not None and tts_engine is not None
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/livekit/test', methods=['POST'])
def test_livekit():
    """Test LiveKit components specifically"""
    try:
        # Check Google credentials status
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "Not set")
        creds_status = {
            'google_creds_ok': google_creds_ok,
            'creds_path': creds_path,
            'creds_file_exists': os.path.exists(creds_path) if creds_path != "Not set" else False,
            'secret_file_exists': os.path.exists("/etc/secrets/google-credentials.json"),
            'local_file_exists': os.path.exists("google-credentials.json"),
            'json_env_var': "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ
        }
        
        # Try to initialize LiveKit components
        livekit_init_result = initialize_livekit_components()
        
        # Test Google STT if available
        stt_test_result = None
        if stt_engine:
            try:
                # This is just a test - we can't actually process audio without a real audio file
                stt_test_result = "STT engine initialized successfully"
            except Exception as e:
                stt_test_result = f"STT test failed: {str(e)}"
        
        # Test Google TTS if available
        tts_test_result = None
        if tts_engine:
            try:
                # This is just a test - we can't actually generate audio without text
                tts_test_result = "TTS engine initialized successfully"
            except Exception as e:
                tts_test_result = f"TTS test failed: {str(e)}"
        
        return jsonify({
            'livekit_initialized': livekit_init_result,
            'google_credentials': creds_status,
            'stt_test': stt_test_result,
            'tts_test': tts_test_result,
            'components': {
                'stt_engine': stt_engine is not None,
                'tts_engine': tts_engine is not None,
                'vad_engine': vad_engine is not None
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/voice-query', methods=['POST'])
def handle_voice_query():
    """Handle voice query using Einstein Agent"""
    data = request.json
    transcript = data.get('text')
    session_id = data.get('session_id', str(uuid.uuid4()))

    if not transcript:
        return jsonify({'error': 'No input text provided'}), 400

    try:
        # Call Einstein Agent asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_text = loop.run_until_complete(call_einstein_agent(transcript, session_id))
        loop.close()

        return jsonify({
            'message': response_text,
            'session_id': session_id
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    """Chat with Einstein Agent with dynamic agent ID"""
    data = request.json
    message = data.get('message')
    session_id = data.get('session_id', str(uuid.uuid4()))
    agent_id = data.get('agent_id')  # Agent ID is required

    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    if not agent_id:
        return jsonify({'error': 'Agent ID is required'}), 400

    try:
        # Call Einstein Agent asynchronously with dynamic agent ID
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_text = loop.run_until_complete(call_einstein_agent(message, session_id, agent_id))
        loop.close()

        return jsonify({
            'response': response_text,
            'session_id': session_id,
            'agent_id': agent_id
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/einstein/agent', methods=['POST'])
def einstein_agent_endpoint():
    """Dedicated Einstein Agent API endpoint with dynamic agent ID"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        session_id = data.get('session_id', str(uuid.uuid4()))
        agent_id = data.get('agent_id')  # Agent ID is required
        
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400
        
        # Call Einstein Agent with dynamic agent ID
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(call_einstein_agent(message, session_id, agent_id))
        loop.close()
        
        return jsonify({
            'response': response,
            'session_id': session_id,
            'agent_id': agent_id
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/process', methods=['POST'])
def process_voice():
    """Process voice input: STT -> Einstein Agent -> TTS"""
    try:
        # Get audio data from request
        audio_file = request.files.get('audio')
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        agent_id = request.form.get('agent_id')  # Agent ID is required
        
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400

        # Process audio with STT
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Convert speech to text
        transcript = loop.run_until_complete(process_voice_with_stt(audio_data))
        
        if not transcript:
            return jsonify({'error': 'Failed to process audio'}), 500
        
        # Get response from Einstein Agent with dynamic agent ID
        response_text = loop.run_until_complete(call_einstein_agent(transcript, session_id, agent_id))
        
        # Convert response to speech
        response_audio = loop.run_until_complete(process_text_with_tts(response_text))
        
        loop.close()

        return jsonify({
            'transcript': transcript,
            'response': response_text,
            'session_id': session_id,
            'agent_id': agent_id,
            'audio_available': response_audio is not None
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using Google TTS"""
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Convert text to speech
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_data = loop.run_until_complete(process_text_with_tts(text))
        loop.close()

        if audio_data:
            return jsonify({
                'message': 'Audio generated successfully',
                'audio_available': True
            }), 200
        else:
            return jsonify({'error': 'Failed to generate audio'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/stt', methods=['POST'])
def speech_to_text():
    """Convert speech to text using Google STT"""
    try:
        # Get audio data from request
        audio_file = request.files.get('audio')
        
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400

        # Process audio with STT
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Convert speech to text
        transcript = loop.run_until_complete(process_voice_with_stt(audio_data))
        loop.close()

        if transcript:
            return jsonify({
                'transcript': transcript,
                'message': 'Speech converted successfully'
            }), 200
        else:
            return jsonify({'error': 'Failed to process audio'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

async def livekit_entrypoint(ctx: JobContext):
    """LiveKit entrypoint for real-time voice processing"""
    logger.info("🚀 Starting Salesforce Voice Agent with LiveKit")
    print("🚀 Starting Salesforce Voice Agent with LiveKit")
    
    # Initialize LiveKit components
    if not initialize_livekit_components():
        logger.error("❌ Failed to initialize LiveKit components")
        return
    
    # Connect to LiveKit room
    logger.info("🔌 Connecting to LiveKit room...")
    print("🔌 Connecting to LiveKit room...")
    await ctx.connect()
    logger.info("✅ Connected to LiveKit room")
    print("✅ Connected to LiveKit room")
    
    # Add user input transcribed handler for real-time voice processing
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        logger.info(f"🎤 User input transcribed: {event.transcript}")
        logger.info(f"Language: {event.language}, Final: {event.is_final}, Speaker ID: {event.speaker_id}")
        
        if event.is_final:  # Only process final transcripts
            transcript = event.transcript
            logger.info(f"🎤 Final transcript captured: {transcript}")
            print(f"🎤 Final transcript captured: {transcript}")
            
            # Call Salesforce API directly with the actual user transcript
            async def call_salesforce_and_respond():
                try:
                    response = await llm_engine._call_salesforce(transcript)
                    logger.info(f"🤖 Salesforce response: {response}")
                    print(f"🤖 Salesforce response: {response}")
                    
                    # Speak the response back to user
                    await agent_session.say(response)
                    logger.info(f"🔊 Response spoken to user: {response}")
                    print(f"🔊 Response spoken to user: {response}")
                except Exception as e:
                    logger.error(f"❌ Error calling Salesforce: {e}")
                    print(f"❌ Error calling Salesforce: {e}")
            
            # Run the async function
            asyncio.create_task(call_salesforce_and_respond())
    
    # Register the user input transcribed handler
    agent_session.on("user_input_transcribed", on_user_input_transcribed)
    
    # Add event handlers
    def on_session_started():
        logger.info("✅ Agent session started")
        print("✅ Agent session started")
    
    def on_agent_output(output):
        logger.info(f"🗣️ Agent said: {output}")
        print(f"🗣️ Agent said: {output}")
    
    def on_tts_start():
        logger.info("🔊 TTS started")
        print("🔊 TTS started")
    
    def on_tts_end():
        logger.info("🔊 TTS ended")
        print("🔊 TTS ended")
    
    # Register event handlers
    agent_session.on("session_started", on_session_started)
    agent_session.on("agent_output", on_agent_output)
    agent_session.on("tts_start", on_tts_start)
    agent_session.on("tts_end", on_tts_end)
    
    # Start the session
    logger.info("🚀 Starting AgentSession...")
    print("🚀 Starting AgentSession...")
    try:
        await agent_session.start(room=ctx.room, agent=agent)
        logger.info("✅ AgentSession started successfully")
        print("✅ AgentSession started successfully")
        
        # Send initial greeting
        logger.info("🔊 Sending initial greeting...")
        print("🔊 Sending initial greeting...")
        
        try:
            await agent_session.say("Hello! I'm your Salesforce voice assistant. I can help you with your Salesforce cases and questions. Please speak into your microphone to start our conversation.")
            logger.info("✅ Initial greeting sent successfully")
            print("✅ Initial greeting sent successfully")
        except Exception as e:
            logger.error(f"❌ Failed to send initial greeting: {e}")
            print(f"❌ Failed to send initial greeting: {e}")
        
        logger.info("🔄 Agent is now running and waiting for voice input...")
        print("🔄 Agent is now running and waiting for voice input...")
        print("🎤 Speak into your microphone to interact with Salesforce!")
        print("🔊 Responses will be spoken back using Google TTS")
        
        # Keep the agent running
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("🛑 Agent shutting down...")
            print("🛑 Agent shutting down...")
            await ctx.shutdown()
        except Exception as e:
            logger.error(f"❌ Unexpected error in agent loop: {e}")
            print(f"❌ Unexpected error in agent loop: {e}")
            raise
            
    except Exception as e:
        logger.error(f"❌ Failed to start AgentSession: {e}")
        print(f"❌ Failed to start AgentSession: {e}")
        raise

if __name__ == '__main__':
    # Check if we should run LiveKit agent or Flask API
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "livekit":
        # Run LiveKit agent
        cli.run_app(WorkerOptions(entrypoint_fnc=livekit_entrypoint))
    else:
        # Run Flask API
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
