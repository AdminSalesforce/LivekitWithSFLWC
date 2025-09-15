# Salesforce Voice Agent with LiveKit

A voice-enabled AI agent that integrates Salesforce Einstein Agent with LiveKit for real-time voice conversations.

## Features

- Real-time voice interaction using LiveKit
- Integration with Salesforce Einstein Agent
- Google Speech-to-Text and Text-to-Speech
- Cloud deployment ready

## Quick Deployment

### 1. Deploy to Render

1. Go to [Render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python salesforce_voice_agent.py`
5. Add environment variables (see below)

### 2. Environment Variables

Set these environment variables in your deployment platform:

```
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=your_livekit_url
GOOGLE_API_KEY=your_google_api_key
SALESFORCE_ORG_DOMAIN=your_salesforce_org_domain
SALESFORCE_CLIENT_ID=your_salesforce_client_id
SALESFORCE_CLIENT_SECRET=your_salesforce_client_secret
SALESFORCE_AGENT_ID=your_salesforce_agent_id
GOOGLE_APPLICATION_CREDENTIALS_JSON=your_google_credentials_json
```

### 3. Test the Agent

Once deployed, your agent will be available at the Render URL. Test it by:

1. Opening the Salesforce LWC components
2. Starting a voice conversation
3. The agent will process your voice and respond through Salesforce Einstein Agent

## Files

- `salesforce_voice_agent.py` - Main voice agent code
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration
- `README.md` - This file

## Support

For issues or questions, check the deployment logs in your Render dashboard.