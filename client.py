#!/usr/bin/env python

"""Client using the asyncio API."""

import asyncio
from websockets.asyncio.client import connect
import os
from dotenv import load_dotenv, dotenv_values

load_dotenv()

async def hello():
    async with connect(f'ws://{os.getenv('HOST')}:{os.getenv('PORT')}') as websocket:
        await websocket.send("Hello world!")
        message = await websocket.recv()
        print(message)


if __name__ == "__main__":
    asyncio.run(hello())