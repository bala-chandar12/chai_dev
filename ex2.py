from flask import Flask, render_template,request
from deepgram import Deepgram
from dotenv import load_dotenv
import os
import asyncio
from aiohttp import web
from aiohttp_wsgi import WSGIHandler
from model import createrag

from typing import Dict, Callable

import time
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

#import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
warnings.filterwarnings("ignore")
#from querygrag import ret
from model import predict

GOOGLE_API_KEY='AIzaSyCvtMa0OoR0OZclO0uC87IV_TlxBkoSv6A'


load_dotenv()

app = Flask('aioflask')

dg_client = Deepgram("012cec2f6ce37fc99e2e035403ac79e71f95ffe3")
dg_client2 = Deepgram("27ce970dc1dac23c32c50c3cc01805e12f6d048a")
question_words = ["who", "what", "when", "where", "why", "how"]


async def process_audio(fast_socket: web.WebSocketResponse):
    async def get_transcript(data: Dict) -> None: 
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']
            transcript_lower=transcript.lower()
            if transcript:
                print("customer:"+transcript)
                for word in question_words:
                    if word in transcript_lower:
                        for i in range(10):
                            print("it is a question")
                        start_index = transcript_lower.find(word)
                        print(transcript)
                        await predict(transcript[start_index:])
                        
                await fast_socket.send_str(transcript)
    deepgram_socket = await connect_to_deepgram(get_transcript)

    return deepgram_socket

async def process_audio2(fast_socket: web.WebSocketResponse):
    async def get_transcript(data: Dict) -> None:
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']
            transcript_lower=transcript.lower()
        
            if transcript:
                print("manager:"+transcript)
                for word in question_words:
                    if word in transcript_lower:
                        for i in range(10):
                            print("it is a question")

                        start_index = transcript_lower.find(word)
                        print(transcript[start_index:])
                        await predict(transcript[start_index:])
                        
                
                
                await fast_socket.send_str(transcript)

    deepgram_socket = await connect_to_deepgram2(get_transcript)

    return deepgram_socket

async def connect_to_deepgram(transcript_received_handler: Callable[[Dict], None]) -> str:
    try:
        socket = await dg_client.transcription.live({'punctuate': True, 'interim_results': False})
        socket.registerHandler(socket.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
        socket.registerHandler(socket.event.TRANSCRIPT_RECEIVED, transcript_received_handler)

        return socket
    except Exception as e:
        raise Exception(f'Could not open socket: {e}')
    
async def connect_to_deepgram2(transcript_received_handler: Callable[[Dict], None]) -> str:
    try:
        socket = await dg_client2.transcription.live({'punctuate': True, 'interim_results': False})
        socket.registerHandler(socket.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
        socket.registerHandler(socket.event.TRANSCRIPT_RECEIVED, transcript_received_handler)

        return socket
    except Exception as e:
        raise Exception(f'Could not open socket: {e}')

@app.route('/')
def index():
    return render_template('index.html')

async def socket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request) 

    deepgram_socket = await process_audio(ws)

    while True:
        data = await ws.receive_bytes()
        deepgram_socket.send(data)

async def socket2(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request) 

    deepgram_socket = await process_audio2(ws)

    while True:
        data = await ws.receive_bytes()
        deepgram_socket.send(data)

async def socket3(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request) 

    deepgram_socket = await process_audio2(ws)

    while True:
        data = await ws.receive_bytes()
        deepgram_socket.send(data)
@app.route('/upload', methods=['POST'])
async def upload(request):
    if request.method == 'POST':
        data = await request.json()
        createrag(data["url"])

    print("hi")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    aio_app = web.Application()
    wsgi = WSGIHandler(app)
    aio_app.router.add_route('*', '/{path_info: *}', wsgi.handle_request)
    aio_app.router.add_route('POST', '/upload', upload)
    aio_app.router.add_route('GET', '/listen', socket)
    aio_app.router.add_route('GET', '/listen2', socket2)
    web.run_app(aio_app, port=5555)