#!/usr/bin/env python3
"""
Salesforce Voice Agent with LiveKit Integration
This agent connects to Salesforce Einstein Agent and provides voice interaction via LiveKit.
"""

import asyncio
import logging
import os
import tempfile
import json
from typing import Optional

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import google
from livekit import rtc

# Enable LiveKit debug logs
os.environ["LIVEKIT_LOG_LEVEL"] = "debug"
logging.getLogger("livekit").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Set environment variables (use environment variables in production)
os.environ["LIVEKIT_API_KEY"] = os.getenv("LIVEKIT_API_KEY", "YOUR_LIVEKIT_API_KEY")
os.environ["LIVEKIT_API_SECRET"] = os.getenv("LIVEKIT_API_SECRET", "YOUR_LIVEKIT_API_SECRET")
os.environ["LIVEKIT_URL"] = os.getenv("LIVEKIT_URL", "YOUR_LIVEKIT_URL")

# Handle Google credentials for cloud deployment
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    # Use JSON string from environment variable (for cloud deployment)
    import tempfile
    import json
    creds_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    creds_data = json.loads(creds_json)
    
    # Create temporary file for Google credentials
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(creds_data, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
else:
    # Use credentials file (for both local and cloud deployment)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google-credentials.json")

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
os.environ["SALESFORCE_ORG_DOMAIN"] = os.getenv("SALESFORCE_ORG_DOMAIN", "YOUR_SALESFORCE_ORG_DOMAIN")
os.environ["SALESFORCE_CLIENT_ID"] = os.getenv("SALESFORCE_CLIENT_ID", "YOUR_SALESFORCE_CLIENT_ID")
os.environ["SALESFORCE_CLIENT_SECRET"] = os.getenv("SALESFORCE_CLIENT_SECRET", "YOUR_SALESFORCE_CLIENT_SECRET")
os.environ["SALESFORCE_AGENT_ID"] = os.getenv("SALESFORCE_AGENT_ID", "YOUR_SALESFORCE_AGENT_ID")


class SalesforceLLM(llm.LLM):
    def __init__(self):
        super().__init__()
        self.last_message = ""
        self.access_token = None
        self.salesforce_domain = os.environ["SALESFORCE_ORG_DOMAIN"]
        self.client_id = os.environ["SALESFORCE_CLIENT_ID"]
        self.client_secret = os.environ["SALESFORCE_CLIENT_SECRET"]
        self.agent_id = os.environ["SALESFORCE_AGENT_ID"]

    async def _get_access_token(self) -> str:
        """Get Salesforce access token using OAuth2 client credentials flow"""
        if self.access_token:
            return self.access_token

        import aiohttp
        
        token_url = f"{self.salesforce_domain}/services/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.access_token = result["access_token"]
                    logger.info("Successfully obtained Salesforce access token")
                    return self.access_token
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get access token: {response.status} - {error_text}")
                    raise Exception(f"Failed to get Salesforce access token: {response.status}")

    async def chat(self, chat_ctx: llm.ChatContext) -> "llm.LLMStream":
        """Chat with Salesforce Einstein Agent"""
        try:
            access_token = await self._get_access_token()
            
            # Prepare the message for Salesforce
            messages = []
            for msg in chat_ctx.messages:
                if msg.role == "user":
                    messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    messages.append({"role": "assistant", "content": msg.content})

            # Call Salesforce Einstein Agent
            response = await self._call_salesforce_agent(access_token, messages)
            
            # Create a simple stream with the response
            return self._create_response_stream(response)
            
        except Exception as e:
            logger.error(f"Error in Salesforce chat: {e}")
            return self._create_error_stream(str(e))

    async def _call_salesforce_agent(self, access_token: str, messages: list) -> str:
        """Call Salesforce Einstein Agent API"""
        import aiohttp
        
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # Call Salesforce Einstein Agent
        agent_url = f"{self.salesforce_domain}/services/data/v58.0/einstein/agent/{self.agent_id}/chat"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": user_message,
            "sessionId": "voice_session_123"  # You might want to generate a unique session ID
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(agent_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "I'm sorry, I couldn't process your request.")
                else:
                    error_text = await response.text()
                    logger.error(f"Salesforce API error: {response.status} - {error_text}")
                    return "I'm sorry, I'm having trouble connecting to Salesforce right now."

    def _create_response_stream(self, response: str) -> "llm.LLMStream":
        """Create a simple response stream"""
        class SimpleStream(llm.LLMStream):
            def __init__(self, text: str):
                self.text = text
                self.done = False

            async def aclose(self):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return llm.LLMStreamChunk(
                    type=llm.LLMStreamChunk.Type.CONTENT,
                    content=response,
                    delta=response
                )

        return SimpleStream(response)

    def _create_error_stream(self, error: str) -> "llm.LLMStream":
        """Create an error response stream"""
        return self._create_response_stream(f"I'm sorry, I encountered an error: {error}")


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent"""
    logger.info("Starting Salesforce Voice Agent")
    
    # Wait for participant to join
    await ctx.wait_for_room()
    participant = ctx.room.local_participant
    logger.info(f"Participant {participant.identity} joined")

    # Simple voice agent implementation
    logger.info("Voice agent is ready and waiting for connections")
    
    # Keep the agent running
    await asyncio.sleep(1)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))