# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from typing import TypedDict

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your Lovable URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. Define the State
class AgentState(TypedDict):
    user_input: str
    visitor_type: str  # Recruiter, Tech Enthusiast, or Casual
    response: str

# 2. Define the Nodes
def analyze_visitor(state: AgentState):
    # Logic to determine who is visiting
    text = state['user_input'].lower()
    if "hire" in text or "job" in text:
        return {"visitor_type": "Recruiter"}
    return {"visitor_type": "Tech Enthusiast"}

def generate_pitch(state: AgentState):
    if state['visitor_type'] == "Recruiter":
        res = "I focus on ROI and scalable AI systems. Check my 'Projects' for production-ready LLMs."
    else:
        res = "Yo! I build cool autonomous agents. Check my GitHub for the LangGraph source code."
    return {"response": res}

# 3. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("analyzer", analyze_visitor)
workflow.add_node("generator", generate_pitch)
workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "generator")
workflow.add_edge("generator", END)

graph = workflow.compile()

# 4. The API Endpoint
@app.post("/chat")
async def chat_endpoint(data: dict):
    inputs = {"user_input": data["message"]}
    result = graph.invoke(inputs)
    return {"reply": result["response"], "type": result["visitor_type"]}

@app.get("/")
async def root():
    return {"status": "Agentic Brain is Online", "version": "1.0"}
