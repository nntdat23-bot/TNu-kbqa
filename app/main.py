import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.schemas import ChatRequest, ChatResponse
from app.agent import run_agent

load_dotenv()

app = FastAPI(title="TNU-AIQA Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "TNU-AIQA KBQA"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = run_agent(request.message)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        model_used=result["model_used"]
    )