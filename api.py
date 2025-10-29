"""
FastAPI 服务：提供问答接口
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agent import QAAgent
from config import load_config
from datetime import datetime
from exception import AgentMissingParamsException, AgentInvalidParamsException


app = FastAPI(title="QA Agent API", version="1.0.0")

# 全局变量：配置和agent实例
config = None
agent = None


class AskRequest(BaseModel):
    """问答请求"""

    question: str
    content_hash: str


class ContextItem(BaseModel):
    """上下文文档项"""

    content: str
    metadata: Optional[dict] = None


class AskResponse(BaseModel):
    """问答响应"""

    question: str
    answer: str
    context: List[ContextItem]
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化配置和agent"""
    global config, agent
    config = load_config()
    agent = QAAgent(config)
    print("[API] QA Agent API started successfully")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "qa-agent",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    回答用户问题

    Args:
        request: 包含问题的请求

    Returns:
        AskResponse: 包含答案和上下文的响应
    """
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")

        print(f"[API] 收到问题: {request.question}")

        # 调用agent处理问题
        state = {"question": request.question, "content_hash": request.content_hash}
        result = await agent.run(state)

        # 构造上下文列表
        context_items = []
        if result.get("context"):
            for doc in result["context"]:
                context_items.append(
                    ContextItem(
                        content=doc.page_content,
                        metadata=doc.metadata if hasattr(doc, "metadata") else None,
                    )
                )

        print(f"[API] 生成答案完成，上下文文档数: {len(context_items)}")
        # print(f"Context: {context_items}")s

        return AskResponse(
            question=request.question,
            answer=result.get("answer", ""),
            context=context_items,
            timestamp=datetime.now().isoformat(),
        )
    except AgentMissingParamsException:
        raise HTTPException(status_code=400, detail="缺少必要参数")
    except AgentInvalidParamsException:
        raise HTTPException(status_code=400, detail="参数无效")
    except Exception as e:
        error_msg = f"处理问题失败: {str(e)}"
        print(f"[API] 错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
