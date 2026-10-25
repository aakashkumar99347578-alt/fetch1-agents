"""
Script to register this agent with Fetch.ai Agentverse.
Run this script locally or during deployment to register the agent.
"""
import os
import sys
from dotenv import load_dotenv

# Ensure console supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "Multi-agent-research-system-main", ".env"))

def register():
    agentverse_key = os.getenv("AGENTVERSE_KEY")
    seed_phrase = os.getenv("AGENT_SEED_PHRASE")
    agent_name = os.getenv("AGENT_NAME", "your researcher dost")
    endpoint = os.getenv("AGENT_ENDPOINT", "https://fetch1-agents.onrender.com")

    print("\n" + "=" * 50)
    print("🤖 Agentverse Agent Registration")
    print("=" * 50)
    print(f"Agent Name: {agent_name}")
    print(f"Endpoint:   {endpoint}")

    if not agentverse_key:
        print("\n❌ Error: AGENTVERSE_KEY is not set.")
        print("Please copy the Agentverse API Key from Agentverse and add it to your .env file:")
        print("AGENTVERSE_KEY=your_key_here")
        return False

    if not seed_phrase:
        print("\n❌ Error: AGENT_SEED_PHRASE is not set.")
        print("Please set AGENT_SEED_PHRASE in your .env file.")
        print("Tip: If Agentverse provided a seed phrase in Step 1, use that exact phrase.")
        print("Otherwise, you can use any unique random phrase, for example:")
        print("AGENT_SEED_PHRASE=your-unique-secret-research-agent-seed-2025")
        return False

    try:
        from uagents_core.utils.registration import (
            register_chat_agent,
            RegistrationRequestCredentials,
            AgentverseRequestError,
        )

        print("\nSending registration request to Agentverse...")
        credentials = RegistrationRequestCredentials(
            agentverse_api_key=agentverse_key,
            agent_seed_phrase=seed_phrase,
        )

        success = register_chat_agent(
            name=agent_name,
            endpoint=endpoint,
            active=True,
            credentials=credentials,
            description="Multi-agent AI research assistant powered by LangGraph, Tavily and Google Gemini.",
            readme="""# Multi-Agent Research System
An autonomous multi-agent research pipeline that searches the web, scrapes in-depth resources, writes comprehensive reports, and critiques findings.
""",
        )

        if success:
            print(f"\n✅ Agent '{agent_name}' successfully registered with Agentverse!")
            print("Now click 'Evaluate my agent's registration' on Agentverse.")
            return True
        else:
            print("\n⚠️ Agentverse returned False during registration.")
            return False

    except ImportError:
        print("\n❌ uagents-core is not installed. Run: pip install uagents-core")
        return False
    except AgentverseRequestError as err:
        print(f"\n❌ Agentverse Registration Error: {err}")
        return False
    except Exception as err:
        print(f"\n❌ Unexpected Error: {err}")
        return False

if __name__ == "__main__":
    register()
