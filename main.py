import asyncio
from agent import QAAgent
from config import load_config


async def test_agent():
    system_config = load_config()
    agent = QAAgent(system_config)
    state = {
        "question": "在执行暗区的时候，万一api数据和游戏内结算数据不一致，怎么办？告诉我具体解决办法",
    }
    response = await agent.run(state)
    print(f"answer: {response['answer']}")
    print(f"context: {response['context']}")


if __name__ == "__main__":
    asyncio.run(test_agent())
