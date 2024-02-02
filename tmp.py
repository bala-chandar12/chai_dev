from flask import Flask, render_template
from deepgram import Deepgram
from dotenv import load_dotenv
import os
import asyncio
from aiohttp import web
from aiohttp_wsgi import WSGIHandler
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, Callable

load_dotenv()

app = Flask('aioflask')

# Define the SQLAlchemy model for transcripts
Base = declarative_base()
class Transcript(Base):
    __tablename__ = 'transcripts'

    id = Column(Integer, primary_key=True)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Configure the database
engine = create_engine('sqlite:///transcripts.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

dg_client = Deepgram(os.getenv('DEEPGRAM_API_KEY'))
dg_client2 = Deepgram(os.getenv('DEEPGRAM_API_KEY2'))

async def process_audio(fast_socket: web.WebSocketResponse):
    async def get_transcript(data: Dict) -> None:
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']
        
            if transcript:
                print("customer:" + transcript)
                await fast_socket.send_str(transcript)
                # Save transcript to the database
                save_transcript(transcript)

    deepgram_socket = await connect_to_deepgram(get_transcript)

    return deepgram_socket

async def process_audio2(fast_socket: web.WebSocketResponse):
    async def get_transcript(data: Dict) -> None:
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']
        
            if transcript:
                print("manager:" + transcript)
                await fast_socket.send_str(transcript)
                # Save transcript to the database
                save_transcript(transcript)

    deepgram_socket = await connect_to_deepgram2(get_transcript)

    return deepgram_socket

def save_transcript(transcript: str):
    # Save transcript to the database
    session = Session()
    session.add(Transcript(content=transcript))
    session.commit()
    session.close()

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

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    aio_app = web.Application()
    wsgi = WSGIHandler(app)
    aio_app.router.add_route('*', '/{path_info: *}', wsgi.handle_request)
    aio_app.router.add_route('GET', '/listen', socket)
    aio_app.router.add_route('GET', '/listen2', socket2)
    web.run_app(aio_app, port=5555)
