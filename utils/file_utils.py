import io

import httpx


async def process_file(file_url: str) -> io.BytesIO | None:
    if file_url.startswith("http"):
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url, timeout=60)
            response.raise_for_status()
            return io.BytesIO(response.content)
    else:
        with open(file_url, "rb") as f:
            content_bytes = f.read()
            return io.BytesIO(content_bytes)
