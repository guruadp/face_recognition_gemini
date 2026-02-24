import os
import re
import requests
from google import genai

SERVICE_BASE_URL = os.getenv("RECOGNIZER_BASE_URL", "http://127.0.0.1:8008").rstrip("/")
RECOGNIZER_URLS = [
    f"{SERVICE_BASE_URL}/identify",
    f"{SERVICE_BASE_URL}/recognize",
    f"{SERVICE_BASE_URL}/recognizer",
]
ENROLL_URL = f"{SERVICE_BASE_URL}/enroll"
ENROLL_SAMPLES = int(os.getenv("ENROLL_SAMPLES", "20"))

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
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
pending_enroll_offer = False
awaiting_enroll_name = False

def should_call_recognition(user_text: str) -> bool:
    t = user_text.lower().strip()
    patterns = [
        r"\bdo you (know\w*|recogniz\w*) me\b",
        r"\bwho am i\b",
        r"\bcan you recogniz\w* me\b",
        r"\bdo you remember me\b",
        r"\bwhat'?s my name\b",
        r"\bwhat is my name\b",
        r"\bidentify me\b",
        r"\bdo you know who i am\b",
    ]
    return any(re.search(p, t) for p in patterns)

def call_recognizer():
    errors = []
    for url in RECOGNIZER_URLS:
        try:
            # CPU inference can be slow on first runs; allow more read time.
            r = requests.get(url, timeout=(1.5, 8.0))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            errors.append(f"{url}: {e}")
    return {"status": "recognizer_error", "error": " | ".join(errors), "name": None, "score": 0.0}

def call_enroll(name: str):
    try:
        payload = {"name": name, "samples": ENROLL_SAMPLES}
        r = requests.post(ENROLL_URL, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "enroll_error", "error": str(e), "name": name}

def is_yes(text: str) -> bool:
    t = text.lower().strip()
    return bool(re.search(r"\b(yes|yeah|yep|sure|ok|okay|please do|go ahead)\b", t))

def is_no(text: str) -> bool:
    t = text.lower().strip()
    return bool(re.search(r"\b(no|nope|nah|not now|don't)\b", t))

def extract_name(text: str) -> str:
    t = text.strip()
    m = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([a-zA-Z][a-zA-Z0-9 _-]{0,62})", t, flags=re.I)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"[^a-zA-Z0-9 _-]", "", t)
    t = " ".join(t.split())
    return t[:64]

def recognition_reply(recog: dict) -> str:
    global pending_enroll_offer, awaiting_enroll_name
    status = (recog or {}).get("status", "recognizer_error")
    name = (recog or {}).get("name")
    score = float((recog or {}).get("score", 0.0))

    if status == "known" and name:
        pending_enroll_offer = False
        awaiting_enroll_name = False
        return f"Yes, I recognize you. Hello {name}."
    if status == "familiar":
        pending_enroll_offer = False
        awaiting_enroll_name = False
        return f"You look familiar to me (score {score:.2f}), but I cannot confirm your name yet."
    if status == "unknown":
        pending_enroll_offer = True
        awaiting_enroll_name = False
        return "I do not recognize you. Do you want me to remember your face?"
    if status == "too_far":
        return "Please move a little closer so I can identify you."
    if status == "no_face":
        return "I cannot see a face clearly right now. Please look at the camera."
    if status == "recognizer_error":
        return "I cannot reach the recognition service right now. Please start recognizer_service.py and try again."
    return "I do not recognize you right now."

def respond(user_text: str) -> str:
    global pending_enroll_offer, awaiting_enroll_name

    if pending_enroll_offer:
        if is_yes(user_text):
            pending_enroll_offer = False
            awaiting_enroll_name = True
            return "Great. What name should I save for your face?"
        if is_no(user_text):
            pending_enroll_offer = False
            awaiting_enroll_name = False
            return "Okay, I will not enroll right now."
        return "Please answer yes or no. Do you want me to remember your face?"

    if awaiting_enroll_name:
        person = extract_name(user_text)
        if not person:
            return "I did not catch the name. Please tell me the name to save."
        result = call_enroll(person)
        awaiting_enroll_name = False
        status = result.get("status")
        if status == "enrolled":
            return f"Done. I enrolled {result.get('name')} with {result.get('samples')} samples."
        if status == "not_enough_samples":
            return (
                f"I could not capture enough samples ({result.get('captured')}/{result.get('required')}). "
                "Please look at the camera and try again."
            )
        if status == "invalid_name":
            return "That name is invalid. Please try a simple name."
        return f"Enrollment failed. {result.get('error', 'Service error')}"

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
