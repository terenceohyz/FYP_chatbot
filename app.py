import os
from operator import itemgetter
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
import chainlit as cl
from langchain_community.llms import HuggingFaceHub
from typing import Dict, Optional
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, ConversationTokenBufferMemory
from chainlit.types import ThreadDict

local_llm = "zephyr-7b-alpha.Q4_K_S.gguf"

config = {
    'context_length': 700,
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 1.2,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2),
}

llm_init = CTransformers(
    model = local_llm,
    model_type = "mistral",
    lib = "avx2",
    config = config
)

# repo_id = "medicalai/ClinicalGPT-base-zh"

# llm_init = HuggingFaceHub(
#     repo_id=repo_id,
#     # task="text-generation",
#     model_kwargs={
#         "max_new_tokens": 250,
#         "top_k": 30,
#         "temperature": 0.1,
#         "repetition_penalty": 1.03,
#     },
#     huggingfacehub_api_token="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",
# )

print(llm_init)

template = """You are a medical chatbot, answer the new question directly with the previous interaction as context.

Previous interaction: {history}

New Question: {question}
Answer: 
"""

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user


async def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    prompt = PromptTemplate(template=template, input_variables = ["history", "question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_init, verbose=True, memory=memory)

    # store chain in user session
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_chat_start
async def main():
    # initiate conversation memory, keeping memory of previous k interactions 
    cl.user_session.set("memory", ConversationBufferWindowMemory(return_messages=True, memory_key="history", k=1))

    await setup_runnable()
    

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="history", k=1)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
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

    # call chain asynchronously
    res = await llm_chain.acall(msg, callbacks=[cl.AsyncLangchainCallbackHandler()])
    actions = [
        cl.Action(name="regenerate", label="regenerate", value="regenerate", description="Regenerate response")
    ]

    msg = cl.user_session.get("prev_message")
    msg.content = res["text"]
    
    await msg.update()
    

@cl.on_message
async def main(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    prev_context = []
    if len(memory.chat_memory.messages) > 0:
        for context in memory.chat_memory.messages:
            prev_context.append(context.content)
    cl.user_session.set("prev_context", prev_context)
    if memory:
        print(memory.chat_memory.messages)

    # retrieve chain from user session
    llm_chain = cl.user_session.get("llm_chain")
    msg = message.content
    cl.user_session.set("prev_user_message", msg)

    # call chain asynchronously
    res = await llm_chain.ainvoke(msg, callbacks=[cl.AsyncLangchainCallbackHandler()])

    actions = [
        cl.Action(name="regenerate", label="regenerate", value="regenerate", description="Regenerate response")
    ]
    msg = cl.Message(content=res["text"], actions=actions)

    prev_msg = cl.user_session.get("prev_message")
    
    cl.user_session.set("prev_message", msg)

    # return result
    await msg.send()
    if prev_msg:
        await prev_msg.remove_actions()