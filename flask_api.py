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

# TTS engine cache to store TTS engines for different voices
tts_engine_cache = {}

# Salesforce configuration
SALESFORCE_ORG_DOMAIN = os.environ.get('SALESFORCE_ORG_DOMAIN', '')
SALESFORCE_CLIENT_ID = os.environ.get('SALESFORCE_CLIENT_ID', '')
SALESFORCE_CLIENT_SECRET = os.environ.get('SALESFORCE_CLIENT_SECRET', '')
SALESFORCE_AGENT_ID = os.environ.get('SALESFORCE_AGENT_ID', '')

# Cache for Salesforce access tokens and sessions
salesforce_token_cache = {}
salesforce_session_cache = {}

def preprocess_text_for_tts(text):
    """Preprocess text to make TTS sound more natural and human-like"""
    if not text:
        return text
    
    # Make text more conversational and natural
    processed_text = text
    
    # Add natural pauses and emphasis for better flow
    processed_text = processed_text.replace('.', '. ')  # Pause after periods
    processed_text = processed_text.replace('!', '! ')  # Pause after exclamations
    processed_text = processed_text.replace('?', '? ')  # Pause after questions
    processed_text = processed_text.replace(',', ', ')  # Pause after commas
    
    # Replace numbers with their spoken equivalents (but keep some context)
    number_replacements = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }
    
    # Only replace standalone numbers, not in words
    import re
    for digit, word in number_replacements.items():
        # Replace digits that are not part of words
        processed_text = re.sub(r'\b' + digit + r'\b', word, processed_text)
    
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
    
    # Add natural emphasis for important words
    emphasis_words = ['important', 'urgent', 'critical', 'error', 'success', 'warning', 'note']
    for word in emphasis_words:
        if word in processed_text.lower():
            processed_text = processed_text.replace(word, f"<emphasis level='strong'>{word}</emphasis>")
            processed_text = processed_text.replace(word.title(), f"<emphasis level='strong'>{word.title()}</emphasis>")
            processed_text = processed_text.replace(word.upper(), f"<emphasis level='strong'>{word.upper()}</emphasis>")
    
    # Clean up multiple spaces
    processed_text = ' '.join(processed_text.split())
    
    print(f"🔧 Text preprocessing for natural speech: '{text[:50]}...' → '{processed_text[:50]}...'")
    return processed_text

def process_text_with_tts_sync(text, language='en-US', voice='en-US-Wavenet-A'):
    """TTS processing using subprocess to isolate async LiveKit TTS (fixes event loop issue)"""
    try:
        print("🔧 process_text_with_tts_sync FUNCTION CALLED")
        print(f"🔧 Original text: {text[:50]}...")
        print(f"🔧 Language: {language}, Voice: {voice}")
        
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
       print(f"🔧 Reading text from file: {temp_file_path}")
       with open('{temp_file_path}', 'r', encoding='utf-8') as f:
           text = f.read()
       
       print(f"🔧 Processing text: {{text[:50]}}...")
       
       if not text.strip():
           print("❌ Empty text provided")
           return None
       
       # Generate audio using LiveKit TTS synthesize method
       print("🔧 Calling tts_engine.synthesize()...")
       audio_stream = tts_engine.synthesize(text=text)
       print("✅ Audio stream created")
       
       audio_chunks = []
       
       # Process audio stream
       print("🔧 Processing audio stream...")
       try:
           async for chunk in audio_stream:
               if hasattr(chunk, 'frame') and chunk.frame:
                   audio_data = chunk.frame.data
                   audio_chunks.append(audio_data)
                   print(f"🔧 Collected chunk: {{len(audio_data)}} bytes")
               elif hasattr(chunk, 'data'):
                   audio_data = chunk.data
                   audio_chunks.append(audio_data)
                   print(f"🔧 Collected chunk (data): {{len(audio_data)}} bytes")
               else:
                   print(f"🔧 Chunk type: {{type(chunk)}}, attributes: {{dir(chunk)}}")
       except Exception as stream_error:
           print(f"❌ Error processing audio stream: {{stream_error}}")
           return None
       
       if not audio_chunks:
           print("❌ No audio chunks collected")
           return None
       
       # Combine audio data
       full_audio_bytes = b"".join(audio_chunks)
       print(f"✅ Total audio bytes: {{len(full_audio_bytes)}}")
       
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
       print(f"✅ WAV audio created: {{len(wav_audio)}} bytes, Base64: {{len(audio_base64)}} chars")
       
       return audio_base64
       
   except Exception as e:
       print(f"❌ Error in TTS generation: {{e}}")
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
        
        # Initialize TTS engine
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
            "stt_engine": "Browser-based (LWC)",
            "tts_engine": tts_engine is not None,
            "agent_session": agent_session is not None,
            "agent": agent is not None
        },
        "salesforce_config": {
            "org_domain": "SET" if SALESFORCE_ORG_DOMAIN else "NOT SET",
            "client_id": "SET" if SALESFORCE_CLIENT_ID else "NOT SET",
            "client_secret": "SET" if SALESFORCE_CLIENT_SECRET else "NOT SET",
            "agent_id": SALESFORCE_AGENT_ID or "NOT SET"
        },
        "cache_status": {
            "has_access_token": "access_token" in salesforce_token_cache,
            "token_expires_at": salesforce_token_cache.get('expires_at', 'N/A'),
            "cached_sessions": list(salesforce_session_cache.keys()),
            "salesforce_sessions": [k for k in salesforce_session_cache.keys() if k.endswith('_salesforce_session')]
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
        provided_session_id = data.get('session_id', '')
        agent_id = data.get('agent_id', SALESFORCE_AGENT_ID or 'agent_001')
        
        print(f"Message: {message}")
        print(f"Provided Session ID: {provided_session_id}")
        print(f"Agent ID: {agent_id}")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Use provided session_id or get/create a persistent one
        if provided_session_id:
            session_id = provided_session_id
            print(f"✅ Using provided session ID: {session_id}")
        else:
            session_id = get_or_create_session_id(agent_id)
            print(f"✅ Using persistent session ID: {session_id}")
        
        # Call actual Salesforce Einstein Agent API
        print("🔧 Calling Salesforce Einstein Agent API...")
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

        # Process text with TTS using LiveKit directly (matching working code)
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
    print("🔧 Starting Flask API server...")
    print("🔧 Flask API will run on port 10000")
    app.run(host='0.0.0.0', port=10000, debug=False)