import asyncio
from agent import QAAgent
from config import load_config


async def test_agent():
    system_config = load_config()
    agent = QAAgent(system_config)
    state = {
        "question": "有一种名为键键豚的海豚吗？",
        "file_type": "txt",
        "file_url": "./files/dolphin.txt",
    }
    response = await agent.run(state)
    print(f"answer: {response['answer']}")
    print(f"context: {response['context']}")


if __name__ == "__main__":
    asyncio.run(test_agent())
