import os
import json
import logging
import traceback
from datetime import datetime
from typing import Optional, Literal, Dict, Any, List

import chromadb
import aiofiles
import mimetypes
import urllib.parse

from fastapi import (FastAPI, Request, HTTPException, File, Form, UploadFile, Body, status)
from fastapi.responses import (StreamingResponse, PlainTextResponse, JSONResponse)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# Import local modules
from engine import model, prompt_template, semantic_search
from add_data import add_into_collection
# Utility Functions
def handle_exception(e: Exception, status_code: int = 500) -> JSONResponse:
    """
    Centralized exception handling utility
    
    Args:
        e (Exception): Caught exception
        status_code (int): HTTP status code
    
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Error: {e}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=status_code, content={"status": "error", "message": str(e), "details": traceback.format_exc()}
    )


def validate_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url (str): URL to validate
    
    Returns:
        bool: Whether URL is valid
    """
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_file_size(file: UploadFile, max_size: int = 50 * 1024 * 1024) -> bool:
    """
    Validate file size
    
    Args:
        file (UploadFile): Uploaded file
        max_size (int): Maximum file size in bytes
    
    Returns:
        bool: Whether file size is acceptable
    """
    return file.size <= max_size if hasattr(file, 'size') else True


def get_file_type(filename: str) -> str:
    """
    Determine file type by extension
    
    Args:
        filename (str): Filename to check
    
    Returns:
        str: File type
    """
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type.split('/')[0]
    return 'unknown'


# Pydantic Models
class StorageCreate(BaseModel):
    name: str
    description: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Storage name must be at least 2 characters long')
        return v.strip()


class StorageUpdate(BaseModel):
    name: str
    description: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Storage name must be at least 2 characters long')
        return v.strip()


class ChatCreate(BaseModel):
    name: str
    model_id: Optional[int] = None


class ChatHistoryCreate(BaseModel):
    chat_id: int
    text: str
    author: Literal['user', 'model']


# ChromaDB Client
chroma_client = chromadb.HttpClient(host='localhost', port=8027)

# FastAPI App
app = FastAPI()

# CORS Configuration
origins = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8027"]

app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"], )


# Async RAG Query Function
async def query_simple_rag_stream(query: str, collection_name: str, chat_id: int):
    try:
        results = semantic_search(query, collection_name)

        context_text = "\n\n---\n\n".join(results['documents'][0])
        sources = list(set([i.get("url", "pdf source") for i in results['metadatas'][0]]))

        system_message_content = prompt_template.format(
            context=context_text, question=query
        )

        response_stream = model.astream(prompt)
        return response_stream, sources

    except Exception as e:
        logger.error(f"RAG Query Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing RAG query: {str(e)}"
        )


# New function for OpenAI model query
async def query_openai_model_stream(query: str, token: str, selected_model: str, chat_id: int):
    try:
        # Initialize the OpenAI model
        llm = ChatOpenAI(
            model=selected_model, api_key=token, timeout=40,
            base_url="https://lk.neuroapi.host/v1"
        )

        # Prepare the context and history
        results = semantic_search(query, "test")
        context_text = "\n\n---\n\n".join(results['documents'][0])
        sources = list(set([i.get("url", "pdf source") for i in results['metadatas'][0]]))

        history = []
        for i in get_last_n_messages(chat_id, 5):
            history.append(
                HumanMessage(i['text']) if i['author'] == 'user' else AIMessage(i['text'])
            )

        # Prepare the prompt
        system_message_content = f"Answer the question based only on the following context:\n{context_text}\n---\n{query}"
        prompt = [{"role": "system", "content": system_message_content}] + [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in history
        ]

        # Generate response
        response = llm.astream(prompt)
        return response, sources

    except Exception as e:
        logger.error(f"OpenAI Query Error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing OpenAI query: {str(e)}"
        )


# Root Endpoint
@app.get('/')
async def root():
    return PlainTextResponse("XaxaTeam")



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='localhost', port=8040)
