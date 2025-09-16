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
from livekit.agents import Agent
from livekit.agents.voice import AgentSession
from livekit.plugins import google

app = Flask(__name__)
CORS(app)

# Enable LiveKit debug logs
os.environ["LIVEKIT_LOG_LEVEL"] = "debug"
logging.getLogger("livekit").setLevel(logging.DEBUG)

# Salesforce configuration
print("🔧 Loading Salesforce configuration...")
SALESFORCE_DOMAIN = os.getenv("SALESFORCE_ORG_DOMAIN", "https://de1740385138027.my.salesforce.com")
SALESFORCE_CLIENT_ID = os.getenv("SALESFORCE_CLIENT_ID")
SALESFORCE_CLIENT_SECRET = os.getenv("SALESFORCE_CLIENT_SECRET")

print(f"🔧 Salesforce Configuration Status:")
print(f"  - Domain: {SALESFORCE_DOMAIN}")
print(f"  - Client ID: {'SET' if SALESFORCE_CLIENT_ID else 'NOT SET'}")
print(f"  - Client Secret: {'SET' if SALESFORCE_CLIENT_SECRET else 'NOT SET'}")

if SALESFORCE_CLIENT_ID:
    print(f"  - Client ID Value: {SALESFORCE_CLIENT_ID[:20]}...")
else:
    print("  - Client ID Value: None")

if SALESFORCE_CLIENT_SECRET:
    print(f"  - Client Secret Value: {SALESFORCE_CLIENT_SECRET[:20]}...")
else:
    print("  - Client Secret Value: None")

# Validate Salesforce configuration
if not SALESFORCE_CLIENT_ID or SALESFORCE_CLIENT_ID == "YOUR_SALESFORCE_CLIENT_ID":
    print("❌ SALESFORCE_CLIENT_ID environment variable not set")
    logger.error("SALESFORCE_CLIENT_ID environment variable not set")
else:
    print("✅ SALESFORCE_CLIENT_ID is properly set")

if not SALESFORCE_CLIENT_SECRET or SALESFORCE_CLIENT_SECRET == "YOUR_SALESFORCE_CLIENT_SECRET":
    print("❌ SALESFORCE_CLIENT_SECRET environment variable not set")
    logger.error("SALESFORCE_CLIENT_SECRET environment variable not set")
else:
    print("✅ SALESFORCE_CLIENT_SECRET is properly set")

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

async def start_salesforce_session(agent_id, session_id):
    """Start a Salesforce Einstein Agent session"""
    try:
        print(f"🔧 Starting Salesforce session:")
        print(f"  - Agent ID: {agent_id}")
        print(f"  - Session ID: {session_id}")
        
        # Get access token first
        access_token = await get_salesforce_access_token()
        if not access_token:
            print("❌ Failed to get Salesforce access token")
            return None
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Use api.salesforce.com for session creation
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/agents/{agent_id}/sessions"
        print(f"🔧 Session creation URL: {url}")
        
        # Match your working payload structure exactly
        payload = {
            "externalSessionKey": session_id,
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
                "chunkTypes": [
                    "Text"
                ]
            },
            "bypassUser": False
        }
        
        print(f"🔧 Session payload: {json.dumps(payload, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                print(f"🔧 Start session response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Session started successfully")
                    print(f"  - Full session response: {json.dumps(result, indent=2)}")
                    
                    # Store the actual session ID from Salesforce response
                    salesforce_session_id = result.get('sessionId')
                    print(f"✅ Salesforce session ID: {salesforce_session_id}")
                    print(f"  - Session ID length: {len(salesforce_session_id) if salesforce_session_id else 0}")
                    print(f"  - Response keys: {list(result.keys())}")
                    return salesforce_session_id
                else:
                    text = await response.text()
                    print(f"❌ Failed to start session: {text}")
                    return None
                    
    except Exception as e:
        print(f"❌ Error starting Salesforce session: {e}")
        import traceback
        traceback.print_exc()
        return None

async def call_einstein_agent(message, session_id, agent_id=None):
    """Call Salesforce Einstein Agent API with dynamic agent ID"""
    try:
        print(f"🔧 Einstein Agent Call Debug:")
        print(f"  - Message: {message}")
        print(f"  - Session ID: {session_id}")
        print(f"  - Agent ID: {agent_id}")
        
        # Agent ID is required - must be provided dynamically
        if not agent_id:
            print("❌ Error: Agent ID is required but not provided")
            return "Error: Agent ID is required but not provided"
        
        # Get Salesforce access token
        print("🔧 Getting Salesforce access token...")
        access_token = await get_salesforce_access_token()
        if not access_token:
            print("❌ Failed to get Salesforce access token")
            return "I'm sorry, I'm having trouble connecting to Salesforce right now."
        
        # Start a Salesforce session first
        print("🔧 Starting Salesforce session...")
        salesforce_session_id = await start_salesforce_session(agent_id, session_id)
        if not salesforce_session_id:
            print("❌ Failed to start Salesforce session")
            return "I'm sorry, I'm having trouble connecting to Salesforce right now."
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Use api.salesforce.com for message sending
        # Use the actual Salesforce session ID, not our local session ID
        url = f"https://api.salesforce.com/einstein/ai-agent/v1/sessions/{salesforce_session_id}/messages"
        print(f"🔧 Message URL: {url}")
        print(f"🔧 Using Salesforce session ID: {salesforce_session_id}")
        
        # Match your working message payload structure exactly
        payload = {
            "message": {
                "sequenceId": str(int(time.time() * 1000)),
                "type": "Text",
                "text": message
            }
        }
        
        print(f"🔧 Message payload: {json.dumps(payload, indent=2)}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                print(f"🔧 Salesforce API response status: {response.status}")
                print(f"🔧 Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Salesforce message response received")
                    print(f"  - Full response: {json.dumps(result, indent=2)}")
                    print(f"  - Response keys: {list(result.keys())}")
                    
                    # Parse the response according to your working structure
                    if 'message' in result:
                        # Direct message response
                        message_text = result.get('message', 'No response message')
                        print(f"✅ Direct message response: {message_text}")
                        return message_text
                    elif 'messages' in result:
                        # Array response format
                        messages = result.get('messages', [])
                        print(f"  - Messages array length: {len(messages)}")
                        
                        if messages and len(messages) > 0:
                            message_obj = messages[0]
                            message_type = message_obj.get('type', '')
                            print(f"  - First message type: {message_type}")
                            print(f"  - First message content: {json.dumps(message_obj, indent=2)}")
                            
                            # Handle different response types
                            if message_type == 'Inform':
                                response_text = message_obj.get('message', 'No response message')
                                print(f"✅ Inform response: {response_text}")
                                return response_text
                            elif message_type == 'Failure':
                                # Handle failure responses
                                print(f"❌ Salesforce returned Failure: {message_obj}")
                                
                                # Check if there's a message field with content
                                if message_obj.get('message'):
                                    return message_obj.get('message')
                                
                                # Check if there are errors with useful information
                                errors = message_obj.get('errors', [])
                                if errors and len(errors) > 0:
                                    error_msg = errors[0]
                                    if "system error occurred" in error_msg.lower():
                                        return "I'm sorry, I'm having trouble connecting to Salesforce right now."
                                    else:
                                        return f"I encountered an issue: {error_msg}"
                                
                                return "I'm sorry, I'm having trouble connecting to Salesforce right now."
                            elif message_type == 'Text':
                                response_text = message_obj.get('text', 'No response text')
                                print(f"✅ Text response: {response_text}")
                                return response_text
                            else:
                                # For any other type, try to get the message field
                                response_text = message_obj.get('message', f'Received response type: {message_type}')
                                print(f"✅ Other response type ({message_type}): {response_text}")
                                return response_text
                        else:
                            print("❌ No messages in response array")
                            return "No messages in response"
                    else:
                        print("❌ No message or messages field in response")
                        return "No response message found"
                elif response.status == 404:
                    error_text = await response.text()
                    print(f"❌ Einstein Agent API not available (404): {error_text}")
                    return "I'm sorry, I'm having trouble connecting to Salesforce right now."
                else:
                    text = await response.text()
                    print(f"❌ Salesforce call failed: {response.status} - {text}")
                    return "I'm sorry, I'm having trouble connecting to Salesforce right now."
                    
    except Exception as e:
        print(f"❌ Error calling Einstein Agent: {e}")
        import traceback
        traceback.print_exc()
        return "I'm sorry, I encountered an error processing your request."

async def get_salesforce_access_token():
    """Get Salesforce access token using OAuth2 Client Credentials flow"""
    try:
        # Check if credentials are properly set
        if not SALESFORCE_CLIENT_ID or not SALESFORCE_CLIENT_SECRET:
            print("❌ Salesforce credentials not properly configured")
            print(f"  - CLIENT_ID set: {bool(SALESFORCE_CLIENT_ID)}")
            print(f"  - CLIENT_SECRET set: {bool(SALESFORCE_CLIENT_SECRET)}")
            return None
        
        auth_url = f"{SALESFORCE_DOMAIN}/services/oauth2/token"
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': SALESFORCE_CLIENT_ID,
            'client_secret': SALESFORCE_CLIENT_SECRET
        }
        
        print(f"🔧 Salesforce Auth Debug:")
        print(f"  - Domain: {SALESFORCE_DOMAIN}")
        print(f"  - Client ID: {SALESFORCE_CLIENT_ID[:20]}...")
        print(f"  - Auth URL: {auth_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=auth_data) as response:
                print(f"🔧 Auth Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    access_token = result["access_token"]
                    print("✅ Successfully obtained Salesforce access token")
                    print(f"  - Token: {access_token[:20]}...")
                    print(f"  - Token length: {len(access_token)}")
                    print(f"  - Full response: {json.dumps(result, indent=2)}")
                    return access_token
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to get access token: {response.status} - {error_text}")
                    print(f"  - Auth URL: {auth_url}")
                    print(f"  - Client ID: {SALESFORCE_CLIENT_ID}")
                    raise Exception(f"Failed to get Salesforce access token: {response.status} - {error_text}")
    except Exception as e:
        print(f"❌ Error getting Salesforce access token: {e}")
        import traceback
        traceback.print_exc()
        return None

async def get_salesforce_agents():
    """Get list of available Einstein Agents from Salesforce"""
    try:
        access_token = await get_salesforce_access_token()
        if not access_token:
            return None
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Use api.salesforce.com for getting agents
        url = "https://api.salesforce.com/einstein/ai-agent/v1/agents"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                print(f"🔧 Agents API Response Status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Found {len(result.get('agents', []))} agents")
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to get agents: {response.status} - {error_text}")
                    return None
                    
    except Exception as e:
        print(f"❌ Error getting Salesforce agents: {e}")
        import traceback
        traceback.print_exc()
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

@app.route('/api/debug/salesforce')
def debug_salesforce():
    """Debug endpoint to check Salesforce credentials setup"""
    return jsonify({
        "salesforce_domain": SALESFORCE_DOMAIN,
        "client_id_set": bool(SALESFORCE_CLIENT_ID),
        "client_secret_set": bool(SALESFORCE_CLIENT_SECRET),
        "client_id_value": SALESFORCE_CLIENT_ID[:20] + "..." if SALESFORCE_CLIENT_ID else None,
        "client_secret_value": SALESFORCE_CLIENT_SECRET[:20] + "..." if SALESFORCE_CLIENT_SECRET else None,
        "environment_variables": {
            "SALESFORCE_ORG_DOMAIN": os.getenv("SALESFORCE_ORG_DOMAIN"),
            "SALESFORCE_CLIENT_ID": "SET" if os.getenv("SALESFORCE_CLIENT_ID") else "NOT SET",
            "SALESFORCE_CLIENT_SECRET": "SET" if os.getenv("SALESFORCE_CLIENT_SECRET") else "NOT SET"
        }
    })

@app.route('/api/test/salesforce-auth')
def test_salesforce_auth():
    """Test endpoint to verify Salesforce authentication"""
    try:
        print("🔧 Testing Salesforce authentication...")
        
        # Test getting access token
        access_token = asyncio.run(get_salesforce_access_token())
        
        if access_token:
            return jsonify({
                "success": True,
                "message": "Salesforce authentication successful",
                "access_token_preview": access_token[:20] + "...",
                "token_length": len(access_token)
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to get Salesforce access token",
                "error": "Authentication failed"
            })
    except Exception as e:
        print(f"❌ Error testing Salesforce auth: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Error testing Salesforce authentication",
            "error": str(e)
        })

@app.route('/api/test/einstein-agent', methods=['POST'])
def test_einstein_agent():
    """Test endpoint to verify Einstein Agent API call"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', '0XxKj000001HwOrKAK')  # Default agent ID
        test_message = data.get('message', 'Hello, this is a test message')
        session_id = data.get('session_id', 'test_session_' + str(int(time.time())))
        
        print(f"🔧 Testing Einstein Agent with:")
        print(f"  - Agent ID: {agent_id}")
        print(f"  - Message: {test_message}")
        print(f"  - Session ID: {session_id}")
        
        # Call Einstein Agent
        result = asyncio.run(call_einstein_agent(test_message, session_id, agent_id))
        
        return jsonify({
            "success": True,
            "message": "Einstein Agent test completed",
            "agent_id": agent_id,
            "session_id": session_id,
            "test_message": test_message,
            "agent_response": result,
            "response_type": type(result).__name__
        })
        
    except Exception as e:
        print(f"❌ Error testing Einstein Agent: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Error testing Einstein Agent",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/api/test/complete-flow', methods=['POST'])
def test_complete_flow():
    """Test the complete flow: Einstein Agent + TTS"""
    try:
        data = request.get_json()
        agent_id = data.get('agent_id', '0XxKj000001HwOrKAK')
        test_message = data.get('message', 'Hello, this is a complete flow test')
        session_id = data.get('session_id', 'complete_test_' + str(int(time.time())))
        
        print(f"🔧 Testing complete flow:")
        print(f"  - Agent ID: {agent_id}")
        print(f"  - Message: {test_message}")
        print(f"  - Session ID: {session_id}")
        
        # Step 1: Call Einstein Agent
        print("🔧 Step 1: Calling Einstein Agent...")
        agent_response = asyncio.run(call_einstein_agent(test_message, session_id, agent_id))
        print(f"✅ Einstein Agent response: {agent_response}")
        
        # Step 2: Initialize LiveKit components if needed
        print("🔧 Step 2: Initializing LiveKit components...")
        if not tts_engine:
            init_result = initialize_livekit_components()
            if not init_result:
                return jsonify({
                    "success": False,
                    "message": "Failed to initialize LiveKit components",
                    "step": "livekit_init"
                })
        
        # Step 3: Generate TTS
        print("🔧 Step 3: Generating TTS...")
        tts_audio = process_text_with_tts_sync(agent_response)
        
        if tts_audio:
            print(f"✅ TTS generated: {len(tts_audio)} characters")
            return jsonify({
                "success": True,
                "message": "Complete flow test successful",
                "agent_id": agent_id,
                "session_id": session_id,
                "test_message": test_message,
                "agent_response": agent_response,
                "tts_generated": True,
                "tts_audio_length": len(tts_audio),
                "tts_audio_preview": tts_audio[:100] + "..." if len(tts_audio) > 100 else tts_audio
            })
        else:
            return jsonify({
                "success": False,
                "message": "TTS generation failed",
                "step": "tts_generation",
                "agent_response": agent_response
            })
            
    except Exception as e:
        print(f"❌ Error in complete flow test: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Error in complete flow test",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.route('/api/salesforce/agents')
def list_salesforce_agents():
    """List available Einstein Agents from Salesforce"""
    try:
        print("🔧 Listing Salesforce agents...")
        agents = asyncio.run(get_salesforce_agents())
        
        if agents:
            return jsonify({
                "success": True,
                "message": "Successfully retrieved agents",
                "agents": agents.get('agents', []),
                "total_count": len(agents.get('agents', []))
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to retrieve agents",
                "agents": [],
                "total_count": 0
            })
            
    except Exception as e:
        print(f"❌ Error listing agents: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Error listing agents",
            "error": str(e),
            "agents": [],
            "total_count": 0
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

def process_text_with_tts_sync(text, language='en-US', voice='en-US-Wavenet-A'):
    """Synchronous wrapper for TTS processing using subprocess approach"""
    try:
        print(f"🔧 Processing TTS for text: {text[:50]}...")
        print(f"Language: {language}, Voice: {voice}")
        
        if not tts_engine:
            print("❌ TTS engine not available")
            return None
        
        # Use subprocess to run TTS in a completely separate process
        import subprocess
        import tempfile
        import sys
        
        # Create a temporary script file for TTS processing
        tts_script = f"""
import asyncio
import sys
import json
import os
import tempfile
import base64
import struct

# Set up the environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "{os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')}"
os.environ["GOOGLE_API_KEY"] = "{os.environ.get('GOOGLE_API_KEY', '')}"

# Import LiveKit components
from livekit.plugins import google

async def process_tts():
    try:
        # Initialize TTS engine
        tts_engine = google.TTS()
        
        # Process text
        audio_stream = tts_engine.synthesize(text="{text}")
        
        audio_chunks = []
        async for chunk in audio_stream:
            if hasattr(chunk, 'frame') and chunk.frame:
                audio_data = chunk.frame.data
                audio_chunks.append(audio_data)
        
        if not audio_chunks:
            return None
            
        full_audio_bytes = b"".join(audio_chunks)
        
        # Convert to WAV format
        wav_audio = create_wav_file(full_audio_bytes)
        audio_base64 = base64.b64encode(wav_audio).decode('utf-8')
        
        return audio_base64
        
    except Exception as e:
        print(f"Error in TTS subprocess: {{e}}", file=sys.stderr)
        return None

def create_wav_file(audio_data, sample_rate=24000, channels=1, bits_per_sample=16):
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + len(audio_data), b'WAVE', b'fmt ', 16, 1,
        channels, sample_rate, sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8, bits_per_sample, b'data', len(audio_data)
    )
    return wav_header + audio_data

if __name__ == "__main__":
    result = asyncio.run(process_tts())
    if result:
        print(result)
    else:
        sys.exit(1)
"""
        
        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(tts_script)
            script_path = f.name
        
        try:
            # Run the script in a subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                audio_base64 = result.stdout.strip()
                print(f"✅ TTS processing completed successfully")
                return audio_base64
            else:
                print(f"❌ TTS subprocess failed: {result.stderr}")
                return None
                
        finally:
            # Clean up the temporary script file
            try:
                os.unlink(script_path)
            except:
                pass
            
    except subprocess.TimeoutExpired:
        print("❌ TTS processing timed out after 30 seconds")
        return None
    except Exception as e:
        print(f"❌ Error in process_text_with_tts_sync: {e}")
        logger.error(f"Error in process_text_with_tts_sync: {e}")
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)