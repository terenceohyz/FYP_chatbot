import os
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from chainlit.server import app as chainlit_app
from chainlit.context import init_ws_context
from chainlit.session import WebsocketSession

# Langchain and Chainlit imports
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
import chainlit as cl
from langchain.memory import ConversationBufferWindowMemory
from chainlit.types import ThreadDict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from base64 import b64encode, b64decode
from os import urandom
from typing import Dict, Optional

app = FastAPI()

# Mount Chainlit as a sub-application
app.mount("/chainlit", chainlit_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/api/ping")
async def ping():
    return {"message": "pong"}

def fake_auth(authorization: str = Header(None)):
    expected_token = "abc"
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token_value = authorization[len("Bearer "):]
    if token_value != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return token_value

@app.get("/api/get-key")
async def get_key(auth: str = Depends(fake_auth)):
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="Encryption key not configured")
    return {"key": key}

@app.get("/api/get-session-id")
async def get_session_id():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        ws_session = WebsocketSession.get_by_id(session_id=session_id)
        if ws_session is None:
            await websocket.close()
            return
        init_ws_context(ws_session)
        while True:
            data = await websocket.receive_text()
            response = process_data(data)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print(f"WebSocket connection closed for session {session_id}")
    except Exception as e:
        print(f"Error in WebSocket connection for session {session_id}: {e}")
        await websocket.close()

# Add LLM configuration and initialization
local_llm = "zephyr-7b-alpha.Q4_K_S.gguf"
config = {
    'context_length': 700,
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.5,
    'top_k': 50,
    'top_p': 0.9,
    'threads': int(os.cpu_count() / 2),
}

llm_init = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",
    config=config
)

# Function to process incoming WebSocket data
def process_data(data):
    decrypted_message = decrypt_message(data, 'your-secret-key')
    response = "Processed message response"
    return response

# Encryption and Decryption functions
def encrypt_message(message, key):
    salt = urandom(16)
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2**14,
        r=8,
        p=1,
        backend=default_backend()
    )
    key = kdf.derive(key.encode())
    iv = urandom(12)
    encryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    ).encryptor()
    encrypted_data = encryptor.update(message.encode()) + encryptor.finalize()
    return b64encode(iv + encryptor.tag + encrypted_data + salt).decode()

def decrypt_message(encrypted_message, key):
    encrypted_message = b64decode(encrypted_message)
    iv = encrypted_message[:12]
    tag = encrypted_message[12:28]
    ciphertext = encrypted_message[28:-16]
    salt = encrypted_message[-16:]
    
    kdf = Scrypt(
        salt=salt,
        length=32,
        n=2**14,
        r=8,
        p=1,
        backend=default_backend()
    )
    key = kdf.derive(key.encode())
    decryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, tag),
        backend=default_backend()
    ).decryptor()
    
    return decryptor.update(ciphertext) + decryptor.finalize()

# Define the template for your prompt
template = """You are a medical chatbot, answer the new question directly and if you don't have an answer, just say that you don't know.

Previous interaction: {history}

New Question: {question}
Answer: 
"""
@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: Dict[str, str], default_user: cl.User) -> Optional[cl.User]:
    return default_user

async def setup_runnable():
    memory = cl.user_session.get("memory")
    prompt = PromptTemplate(template=template, input_variables=["history", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_init, verbose=True, memory=memory)
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_chat_start
async def main():
    print("Session id:", cl.user_session.get("id"))
    cl.user_session.set("memory", ConversationBufferWindowMemory(return_messages=True, memory_key="history", k=1))
    await setup_runnable()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="history", k=1)
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    message = "How are you feeling now?"
    memory.chat_memory.add_ai_message(message)
    cl.user_session.set("memory", memory)
    await cl.Message(message).send()
    await setup_runnable()

@cl.action_callback("regenerate")
async def on_action(regenerate):
    llm_chain = cl.user_session.get("llm_chain")
    msg = cl.user_session.get("prev_user_message")
    prev_context = cl.user_session.get("prev_context")
    memory = cl.user_session.get("memory")
    if len(prev_context) > 0:
        memory.chat_memory.add_user_message(prev_context[0])
        memory.chat_memory.add_ai_message(prev_context[1])
    else:
        memory.chat_memory.clear()

    res = await llm_chain.acall(msg, callbacks=[cl.AsyncLangchainCallbackHandler()])
    actions = [
        cl.Action(name="regenerate", label="regenerate", value="regenerate", description="Regenerate response")
    ]
    msg = cl.user_session.get("prev_message")
    msg.content = res["text"]
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    encrypted_message = encrypt_message(message.content, 'your-secret-key')
    memory = cl.user_session.get("memory")
    prev_context = [context.content for context in memory.chat_memory.messages] if memory.chat_memory.messages else []
    cl.user_session.set("prev_context", prev_context)
    llm_chain = cl.user_session.get("llm_chain")
    decrypted_message = decrypt_message(encrypted_message, 'your-secret-key')
    cl.user_session.set("prev_user_message", decrypted_message)
    res = await llm_chain.acall(decrypted_message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    encrypted_response = encrypt_message(res["text"], 'your-secret-key')
    actions = [
        cl.Action(name="regenerate", label="regenerate", value="regenerate", description="Regenerate response")
    ]
    msg = cl.Message(content=encrypted_response, actions=actions)
    prev_msg = cl.user_session.get("prev_message")
    cl.user_session.set("prev_message", msg)
    if prev_msg:
        await prev_msg.remove_actions()
    print("Sending encrypted response:", msg)
    await msg.send()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)