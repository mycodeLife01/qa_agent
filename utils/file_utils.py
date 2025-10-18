import io

import httpx


async def process_file(file_url: str) -> io.BytesIO | None:
    async with httpx.AsyncClient() as client:
        response = await client.get(file_url, timeout=60)
        response.raise_for_status()
        return io.BytesIO(response.content)
    