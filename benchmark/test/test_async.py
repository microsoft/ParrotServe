import asyncio


async def hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")


async def stuck():
    while True:
        await asyncio.sleep(1)


async def main():
    coros = [hello(), stuck()]
    await asyncio.gather(*coros)


asyncio.run(main())
