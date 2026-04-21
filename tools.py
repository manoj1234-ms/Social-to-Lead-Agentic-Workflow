from langchain_core.tools import tool

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Call this tool ONLY when you have collected the user's name, email, and creator platform. Use this for high-intent leads."""
    print(f"\n✅ [SYSTEM] Lead captured successfully: {name}, {email}, {platform}\n")
    return "Lead successfully captured."
