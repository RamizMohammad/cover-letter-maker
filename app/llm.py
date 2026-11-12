"""LLM call: one request in, one JSON object out.

Two providers, picked with LLM_PROVIDER:
  anthropic  -> Anthropic Messages API
  openai     -> any OpenAI-compatible /chat/completions endpoint
                (OpenAI, Gemini's compat endpoint, OpenRouter, Ollama, vLLM ...)
The model must accept images if you want to use JD screenshots.
"""
import json
import os

import httpx

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5" if PROVIDER == "anthropic" else "")
BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.anthropic.com/v1" if PROVIDER == "anthropic" else "https://api.openai.com/v1",
).rstrip("/")
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

LENGTHS = {
    "short": "150 to 200 words, 3 short paragraphs",
    "standard": "250 to 350 words, 3 to 4 paragraphs",
    "detailed": "380 to 450 words, 4 to 5 paragraphs",
}
TONES = {
    "professional": "professional and direct, confident without being stiff",
    "conversational": "warm and conversational, like a thoughtful email to a future teammate",
    "enthusiastic": "energetic and keen, but still specific and never gushing",
}

SYSTEM = """You write cover letters for one candidate. You are given their resume text and a job
description (as text, as screenshots, or both). Read screenshots carefully; they are the JD.

Hard rules:
- Use only facts that appear in the resume. Never invent employers, titles, dates, numbers,
  skills, degrees or achievements. If the JD asks for something the resume does not show,
  do not claim it; either leave it out or point to the closest real, adjacent experience.
- Be specific: name real projects, tools and results from the resume that map to the JD's
  most important requirements. Two or three strong matches beat a list of everything.
- No cliches ("I am writing to express my interest", "passionate", "team player",
  "fast-paced environment"), no restating the resume line by line, no bullet points.
- Open with something concrete about the role or company taken from the JD.
- Address a named person only if the JD names one; otherwise "Dear Hiring Manager,".
- Plain text only. Plain ASCII punctuation (straight quotes, hyphens). No markdown.
- The letter starts at the greeting and ends with a sign-off and the candidate's name.
  Do not include addresses, the date, or a subject line.

Return ONLY a JSON object, no prose around it, with exactly this shape:
{
  "company": "company name from the JD, or empty string if not stated",
  "role": "job title from the JD",
  "candidate": {"name": "", "email": "", "phone": "", "links": ""},
  "match": [
    {"requirement": "short JD requirement",
     "status": "covered" | "partial" | "missing",
     "evidence": "one line: what in the resume supports this, or what is absent"}
  ],
  "letter": "the full letter, paragraphs separated by a blank line"
}
"match" lists the 6 to 8 requirements that matter most in the JD, most important first,
judged honestly against the resume. Candidate fields come from the resume; use an empty
string for anything not present ("links" is e.g. LinkedIn / GitHub / portfolio, comma separated)."""


class LLMError(Exception):
    pass


def build_user_text(resume_text: str, jd_text: str, n_images: int, tone: str, length: str) -> str:
    parts = [
        f"Tone: {TONES.get(tone, TONES['professional'])}.",
        f"Length of the letter: {LENGTHS.get(length, LENGTHS['standard'])}.",
        "",
        "<resume>",
        resume_text.strip(),
        "</resume>",
        "",
    ]
    if jd_text.strip():
        parts += ["<job_description>", jd_text.strip(), "</job_description>"]
    if n_images:
        parts.append(f"The job description is also in the {n_images} attached image(s).")
    return "\n".join(parts)


def _anthropic(user_text: str, images: list[tuple[str, str]]) -> str:
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": mt, "data": b64}}
        for mt, b64 in images
    ]
    content.append({"type": "text", "text": user_text})
    r = httpx.post(
        f"{BASE_URL}/messages",
        headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
        json={
            "model": MODEL,
            "max_tokens": 3000,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=TIMEOUT,
    )
    _raise(r)
    return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")


def _openai(user_text: str, images: list[tuple[str, str]]) -> str:
    content = [{"type": "text", "text": user_text}]
    content += [
        {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b64}"}}
        for mt, b64 in images
    ]
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    r = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": content},
            ],
        },
        timeout=TIMEOUT,
    )
    _raise(r)
    return r.json()["choices"][0]["message"]["content"] or ""


def _raise(r: httpx.Response) -> None:
    if r.status_code >= 400:
        raise LLMError(f"{PROVIDER} API returned {r.status_code}: {r.text[:400]}")


def parse_json(raw: str) -> dict:
    """Models sometimes wrap JSON in fences or a sentence; take the outermost object."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("Model did not return JSON.")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Model returned invalid JSON: {e}") from e


def generate(resume_text: str, jd_text: str, images: list[tuple[str, str]], tone: str, length: str) -> dict:
    if not MODEL:
        raise LLMError("LLM_MODEL is not set.")
    if PROVIDER == "anthropic" and not API_KEY:
        raise LLMError("LLM_API_KEY is not set.")
    user_text = build_user_text(resume_text, jd_text, len(images), tone, length)
    try:
        raw = _anthropic(user_text, images) if PROVIDER == "anthropic" else _openai(user_text, images)
    except httpx.HTTPError as e:
        raise LLMError(f"Could not reach the LLM API: {e}") from e
    data = parse_json(raw)
    if not str(data.get("letter", "")).strip():
        raise LLMError("Model returned no letter.")
    data.setdefault("company", "")
    data.setdefault("role", "")
    data.setdefault("match", [])
    cand = data.get("candidate") if isinstance(data.get("candidate"), dict) else {}
    data["candidate"] = {k: str(cand.get(k, "") or "") for k in ("name", "email", "phone", "links")}
    return data
