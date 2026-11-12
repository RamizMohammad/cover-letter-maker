import base64
import io
import os
import re
import secrets
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from pypdf import PdfReader

from . import llm, pdf

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESUME_PATH = DATA_DIR / "resume.pdf"
RESUME_NAME = DATA_DIR / "resume.name"
STATIC = Path(__file__).parent / "static"

APP_USER = os.getenv("APP_USER", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

MAX_PDF = 10 * 1024 * 1024
MAX_IMG = 5 * 1024 * 1024
MAX_IMAGES = 6
IMG_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}

basic = HTTPBasic(auto_error=False)


async def call_health_every_30_seconds():
    while True:
        healthz()
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(_: FastAPI):
    health_task = asyncio.create_task(call_health_every_30_seconds())
    try:
        yield
    finally:
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass


def auth(creds: HTTPBasicCredentials | None = Depends(basic)) -> None:
    """Optional HTTP Basic auth; on when APP_PASSWORD is set."""
    if not APP_PASSWORD:
        return
    ok = (
        creds is not None
        and secrets.compare_digest(creds.username, APP_USER)
        and secrets.compare_digest(creds.password, APP_PASSWORD)
    )
    if not ok:
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Basic"})


app = FastAPI(title="Cover Letter Maker", dependencies=[Depends(auth)], lifespan=lifespan)


def resume_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        raise HTTPException(400, f"Could not read the resume PDF: {e}")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 200:
        raise HTTPException(400, "No selectable text found in the resume PDF (is it a scan?). Export it as a text PDF.")
    return text


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/resume")
def resume_status():
    stored = RESUME_PATH.exists()
    return {"stored": stored, "name": RESUME_NAME.read_text() if stored and RESUME_NAME.exists() else ""}


@app.delete("/api/resume")
def resume_delete():
    RESUME_PATH.unlink(missing_ok=True)
    RESUME_NAME.unlink(missing_ok=True)
    return {"stored": False}


@app.post("/api/generate")
def generate(
    resume: UploadFile | None = File(None),
    jd_text: str = Form(""),
    jd_images: list[UploadFile] = File(default=[]),
    tone: str = Form("professional"),
    length: str = Form("standard"),
):
    # Resume: use the upload (and remember it), otherwise the stored one.
    if resume is not None and resume.filename:
        data = resume.file.read(MAX_PDF + 1)
        if len(data) > MAX_PDF:
            raise HTTPException(413, "Resume PDF is larger than 10 MB.")
        text = resume_text(data)  # validate before saving
        RESUME_PATH.write_bytes(data)
        RESUME_NAME.write_text(Path(resume.filename).name[:120])
    elif RESUME_PATH.exists():
        text = resume_text(RESUME_PATH.read_bytes())
    else:
        raise HTTPException(400, "Upload your resume PDF.")

    images = []
    for f in [f for f in jd_images if f.filename][:MAX_IMAGES]:
        if f.content_type not in IMG_TYPES:
            raise HTTPException(400, f"{f.filename}: only PNG, JPEG, WebP or GIF images are supported.")
        raw = f.file.read(MAX_IMG + 1)
        if len(raw) > MAX_IMG:
            raise HTTPException(413, f"{f.filename} is larger than 5 MB.")
        images.append((f.content_type, base64.b64encode(raw).decode()))

    if not jd_text.strip() and not images:
        raise HTTPException(400, "Paste the job description or add a screenshot of it.")

    try:
        return llm.generate(text, jd_text, images, tone, length)
    except llm.LLMError as e:
        raise HTTPException(502, str(e))


class PdfRequest(BaseModel):
    letter: str
    name: str = ""
    contact: list[str] = []
    filename: str = "cover-letter"


@app.post("/api/pdf")
def make_pdf(req: PdfRequest):
    if not req.letter.strip():
        raise HTTPException(400, "Letter is empty.")
    fname = re.sub(r"[^A-Za-z0-9._-]+", "-", req.filename).strip("-")[:80] or "cover-letter"
    return Response(
        pdf.render(req.letter, req.name, req.contact),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}.pdf"'},
    )



@app.get("/healthz")
def healthz():
    return {"ok": True, "provider": llm.PROVIDER, "model": llm.MODEL}
