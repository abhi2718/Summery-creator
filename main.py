from fastapi import FastAPI, UploadFile, File
from typing import Dict
from transformers import pipeline
import tempfile, os
from faster_whisper import WhisperModel

app = FastAPI()

# ---------- 1. Transcription Model ----------
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# ---------- 2. Hugging Face Pipelines ----------
# ---------- FLAN-T5 pipeline for instruction-based summarization ----------
note_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Store last generated notes for Q&A
NOTES_STORE = {"notes": ""}


@app.get("/")
def root():
    return {"status": "Running"}


# ---------- 3. Transcription ----------
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        content = await file.read()
        temp_audio.write(content)
        temp_audio.flush()
        temp_audio_path = temp_audio.name

    segments, info = whisper_model.transcribe(temp_audio_path)
    full_text = " ".join([segment.text for segment in segments])
    os.remove(temp_audio_path)

    return {"language": info.language, "text": full_text}


# ---------- 4. Summarization ----------
@app.post("/summarize/")
async def summarize_text(payload: Dict) -> Dict:
    text = payload.get("text", "")
    if not text:
        return {"error": "No text provided"}
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    return {"summary": summary}


# ---------- 5. Generate Notes ----------
@app.post("/generate_notes/")
async def generate_notes(payload: Dict) -> Dict:
    text = payload.get("text", "")
    if not text:
        return {"error": "No text provided"}

    # Instruction prompt for bullet-style notes
    prompt = f"Summarize the following lecture transcript into concise, bullet-style study notes: {text}"

    # Generate notes using pipeline
    result = note_pipeline(text, do_sample=False)
    notes = result[0]['generated_text']

    # Save notes for Q&A
    NOTES_STORE["notes"] = notes
    return {"notes": notes}


# ---------- 6. Question Answering ----------
@app.post("/ask/")
async def ask_question(payload: Dict) -> Dict:
    question = payload.get("question", "")
    notes = payload.get("notes","")
    if not notes:
        return {"error": "No notes available. Generate notes first."}
    if not question:
        return {"error": "No question provided."}

    answer = qa_pipeline(question=question, context=notes)
    return {"question": question, "answer": answer['answer']}




