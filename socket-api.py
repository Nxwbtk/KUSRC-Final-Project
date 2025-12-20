import asyncio
import websockets
from websockets.asyncio.server import serve
import os
from dotenv import load_dotenv, dotenv_values
from websockets.asyncio.client import connect
load_dotenv()
from rich import print
class SocketAPI:
    def __init__(self):
        self.host = os.getenv('HOST')
        self.port = os.getenv('PORT')
    
    async def echo(self, websocket):
        try:
            async for message in websocket:
                await websocket.send(message)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    async def run(self):
        print(f"Starting server at ws://{self.host}:{self.port}")
        async with serve(self.echo, self.host, self.port) as server:
            await server.serve_forever()


if __name__ == '__main__':
    socket = SocketAPI()
    asyncio.run(socket.run())