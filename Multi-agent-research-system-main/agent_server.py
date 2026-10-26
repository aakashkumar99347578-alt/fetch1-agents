"""
Agentverse Chat Webhook & Message Server.
Listens for incoming messages from Agentverse, executes the multi-agent
research pipeline, and returns the signed response envelope.
"""
import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from uuid import uuid4
from dotenv import load_dotenv

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route
import uvicorn

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_server")

from uagents_core.identity import Identity
from uagents_core.envelope import Envelope
from uagents_core.models import Model
from uagents_core.protocol import ProtocolSpecification
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    TextContent,
    ChatAcknowledgement,
    chat_protocol_spec,
)
from uagents_core.utils.messages import (
    generate_message_envelope,
    parse_envelope,
    send_message_to_agent,
)
from uagents_core.utils.registration import (
    register_chat_agent,
    RegistrationRequestCredentials,
)

from pipeline import run_research_pipeline

# Initialize agent identity from seed phrase
seed_phrase = os.getenv("AGENT_SEED_PHRASE", "aakash-researcher-agent-2025-a89b")
agent_identity = Identity.from_seed(seed_phrase, 0)
agent_name = os.getenv("AGENT_NAME", "your researcher dost")
endpoint = os.getenv("AGENT_ENDPOINT", "https://fetch1-agents.onrender.com")

logger.info(f"Agent Initialized: {agent_name} | Address: {agent_identity.address}")

# ── Auto-Registration on Startup ──────────────────────────────────────────────
def ensure_agentverse_registered():
    agentverse_key = os.getenv("AGENTVERSE_KEY")
    if not agentverse_key:
        logger.warning("AGENTVERSE_KEY is not set. Skipping Agentverse auto-registration.")
        return False
    try:
        logger.info("Registering agent with Agentverse...")
        creds = RegistrationRequestCredentials(
            agentverse_api_key=agentverse_key,
            agent_seed_phrase=seed_phrase,
        )
        success = register_chat_agent(
            name=agent_name,
            endpoint=endpoint,
            active=True,
            credentials=creds,
            description="Autonomous multi-agent research assistant powered by Google Gemini, Tavily, and LangGraph.",
            readme="""# Multi-Agent Research System
An autonomous multi-agent research pipeline that searches the web, scrapes in-depth resources, writes comprehensive reports, and critiques findings.
""",
        )
        if success:
            logger.info("Agent successfully registered and active on Agentverse!")
        return success
    except Exception as e:
        logger.error(f"Agentverse registration error: {e}")
        return False


# ── Webhook Handler for Agentverse Messages ──────────────────────────────────
async def handle_agentverse_message(request):
    """
    Receives incoming Envelope from Agentverse, extracts the user query,
    executes the research pipeline, and returns a signed reply Envelope.
    """
    try:
        body_bytes = await request.body()
        if not body_bytes:
            return JSONResponse({"status": "ok", "message": "No content received"})

        data = json.loads(body_bytes.decode("utf-8"))
        envelope = Envelope.model_validate(data)
        logger.info(f"Received message from: {envelope.sender} | Session: {envelope.session}")

        # Parse the message inside the envelope
        parsed_msg = parse_envelope(envelope, [ChatMessage, ChatAcknowledgement])

        # If it's a ChatMessage
        user_query = ""
        if isinstance(parsed_msg, ChatMessage):
            user_query = parsed_msg.text().strip()
        elif isinstance(parsed_msg, str):
            try:
                raw_json = json.loads(parsed_msg)
                user_query = raw_json.get("text", "")
            except Exception:
                user_query = str(parsed_msg)

        logger.info(f"User Query: '{user_query}'")

        # Determine response text
        if not user_query or user_query.lower() in ("hi", "hello", "hey", "start", "/start"):
            reply_text = (
                f"Hello! I am **{agent_name}**, your autonomous AI research assistant. 🔬\n\n"
                "Give me any research topic (e.g. *'Quantum computing breakthroughs 2025'* or *'CRISPR gene editing updates'*), "
                "and my agent team will search the web, scrape sources, draft a report, and critique the findings for you!"
            )
        else:
            logger.info(f"Executing research pipeline for topic: '{user_query}'...")
            try:
                pipeline_state = run_research_pipeline(user_query)
                report = pipeline_state.get("report", "No report generated.")
                feedback = pipeline_state.get("feedback", "")

                reply_text = (
                    f"## 📝 Research Report: {user_query}\n\n"
                    f"{report}\n\n"
                    f"---\n"
                    f"### 🧐 Critic Evaluation\n"
                    f"{feedback}"
                )
            except Exception as pipeline_err:
                logger.error(f"Pipeline error: {pipeline_err}")
                reply_text = f"⚠️ An error occurred while researching '{user_query}': {str(pipeline_err)}"

        # Prepare signed reply ChatMessage & Envelope
        reply_msg = ChatMessage(content=[TextContent(text=reply_text)])
        reply_digest = Model.build_schema_digest(reply_msg)
        proto_digest = ProtocolSpecification.compute_digest(chat_protocol_spec.manifest())

        reply_envelope = generate_message_envelope(
            destination=envelope.sender,
            message_schema_digest=reply_digest,
            message_body=json.loads(reply_msg.model_dump_json()),
            sender=agent_identity,
            session_id=envelope.session,
            protocol_digest=proto_digest,
        )

        # Send asynchronously via Almanac if needed
        # Try async send in background thread without blocking direct HTTP response
        import threading
        def _bg_send():
            try:
                send_message_to_agent(
                    destination=envelope.sender,
                    msg=reply_msg,
                    sender=agent_identity,
                    session_id=envelope.session,
                    track_interaction=False,
                )
            except Exception:
                pass
        threading.Thread(target=_bg_send, daemon=True).start()

        # Return the signed envelope directly in the HTTP response
        return JSONResponse(json.loads(reply_envelope.model_dump_json()))

    except Exception as e:
        logger.error(f"Error handling request: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Home / Status Page ────────────────────────────────────────────────────────
async def home(request):
    if request.method == "POST":
        return await handle_agentverse_message(request)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{agent_name} · Agentverse Server</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #0a0a0f;
                color: #e8e4dc;
                margin: 0;
                padding: 40px 20px;
                display: flex;
                justify-content: center;
            }}
            .card {{
                max-width: 650px;
                width: 100%;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,140,50,0.25);
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            }}
            h1 {{ color: #ff8c32; margin-top: 0; font-size: 1.8rem; }}
            .status {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(80,200,120,0.15);
                color: #50c878;
                border: 1px solid rgba(80,200,120,0.3);
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 20px;
            }}
            .dot {{
                width: 8px; height: 8px; background: #50c878; border-radius: 50%;
                box-shadow: 0 0 8px #50c878;
            }}
            .info-box {{
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 10px;
                padding: 16px;
                margin: 16px 0;
                font-family: monospace;
                font-size: 0.85rem;
                word-break: break-all;
            }}
            .label {{ color: #a09890; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }}
            p {{ line-height: 1.6; color: #cdc8bf; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="status"><span class="dot"></span> Online & Listening</div>
            <h1>🤖 {agent_name}</h1>
            <p>This server handles live messaging and research automation for Fetch.ai <strong>Agentverse</strong>.</p>
            
            <div class="label">Agent Address</div>
            <div class="info-box">{agent_identity.address}</div>

            <div class="label">Webhook Endpoint</div>
            <div class="info-box">{endpoint}</div>

            <div class="label">Connected Architecture</div>
            <p>✅ LangGraph Multi-Agent Loop<br>
            ✅ Tavily Web Search Tool<br>
            ✅ Deep Web Scraper Tool<br>
            ✅ Google Gemini Synthesis & Critic Review</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html_content)


# ── Starlette Application Setup ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    try:
        ensure_agentverse_registered()
    except Exception as err:
        logger.error(f"Startup registration error: {err}")
    yield

routes = [
    Route("/", endpoint=home, methods=["GET", "POST"]),
    Route("/webhook", endpoint=handle_agentverse_message, methods=["POST"]),
    Route("/submit", endpoint=handle_agentverse_message, methods=["POST"]),
]

app = Starlette(debug=False, routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
