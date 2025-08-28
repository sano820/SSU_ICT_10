from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.chatbot import ChatRequest, ChatResponse
from app.services.gemini import run_chat_task

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


@router.post("/query", response_model=ChatResponse)
async def chatbot(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    result = await run_chat_task(db, task=req.task, user_input=req.input, context=req.context)
    return result
