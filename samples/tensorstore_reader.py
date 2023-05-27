import asyncio
import random

from aiostream import stream, pipe
from unsync import unsync

from hemera.path_translator import get_path
from eumetsat.datasets import EMTensorstoreDataset
import numpy as np

path = get_path("data") / "EUMETSAT" / "UK-EXT" / "img_z=1514764800,e=1515110340,f=3600.ts"
ext = EMTensorstoreDataset(path)

indexs = np.array((500,500))

items = len(ext)
random.seed(42)
smax2 = 32//2
window = 4

def get_xy():
    return random.randint(0+smax2, 500-smax2)

def get_t():
    return random.randint(0, items-window)

def gen():
    for i in range(200):
        ts = get_t()
        ti = i % items
        ts = slice(ti, ti+1)
        xs = get_xy()
        xs = slice(xs-smax2, xs+smax2)
        ys = get_xy()
        ys = slice(ys-smax2, ys+smax2)

        yield (ts, xs, ys)

async def read_img(sidx):
    s = random.randint(1,10)
    await asyncio.sleep(s)
    return sidx[0].start, s


async def fun():
    st = stream.iterate(gen())
    p = st | pipe.map(read_img, ordered=False, task_limit=5) | pipe.chunks(10)

    async with p.stream() as s:
        async for x in s:
            yield x

async def run():
    async for x in fun():
        print(x)

# def run():
#     for x in a:
#         print(x)
ait = fun()
r1 = asyncio.run(anext(ait))
r2 = asyncio.run(anext(ait))
loop = asyncio.new_event_loop()
fn = fun()
fn_n = anext(fn)
r1 = loop.create_task(fn_n)
res = asyncio.create_task(fn)
i = 0