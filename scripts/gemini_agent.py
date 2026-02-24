import os
import re
import requests
from google import genai

RECOGNIZER_URL = "http://127.0.0.1:8008/identify"

def load_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GEMINI_API_KEY not found in environment or scripts/.env")

client = genai.Client(api_key=load_api_key())
MODEL_NAME = "gemini-2.5-flash"

def should_call_recognition(user_text: str) -> bool:
    t = user_text.lower().strip()
    patterns = [
        r"\bdo you (know|recognize) me\b",
        r"\bwho am i\b",
        r"\bcan you recognize me\b",
        r"\bdo you remember me\b",
        r"\bwhat'?s my name\b",
        r"\bidentify me\b",
    ]
    return any(re.search(p, t) for p in patterns)

def call_recognizer():
    try:
        r = requests.get(RECOGNIZER_URL, timeout=2.5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "recognizer_error", "error": str(e), "name": None, "score": 0.0}

def recognition_reply(recog: dict) -> str:
    status = (recog or {}).get("status", "recognizer_error")
    name = (recog or {}).get("name")
    score = float((recog or {}).get("score", 0.0))

    if status == "known" and name:
        return f"Yes, I recognize you. Hello {name}."
    if status == "familiar":
        return f"You look familiar to me (score {score:.2f}), but I cannot confirm your name yet."
    if status == "unknown":
        return "I do not recognize you yet. I can help you enroll."
    if status == "too_far":
        return "Please move a little closer so I can identify you."
    if status == "no_face":
        return "I cannot see a face clearly right now. Please look at the camera."
    return "I cannot reach the recognition service right now."

def respond(user_text: str) -> str:
    if should_call_recognition(user_text):
        return recognition_reply(call_recognizer())

    prompt = f"""
You are a humanoid reception robot.
Be concise (1–2 sentences). Friendly but not creepy.

User said: {user_text}
"""

    try:
        out = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = out.text or ""
        return text.strip() if text else "I could not generate a response right now."
    except Exception as e:
        return f"I could not reach Gemini right now ({e})."

if __name__ == "__main__":
    print("Type messages. Ctrl+C to quit.")
    while True:
        user = input("\nYou: ").strip()
        if not user:
            continue
        print("Bot:", respond(user))
