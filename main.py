import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain & LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

app = FastAPI()

# 1. CORS SETUP - This allows your Lovable frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you can change this to your Lovable URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DEFINE TOOLS - This makes the agent "Autonomous"
@tool
def get_my_github_projects():
    """
    Fetches the latest repository names from my GitHub. 
    Use this when someone asks what I have built or what my projects are.
    """
    # Replace 'YOUR_GITHUB_USERNAME' with your actual username
    username = "Akshu24Tech" 
    try:
        url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5"
        response = requests.get(url)
        if response.status_code == 200:
            repos = response.json()
            repo_list = [f"{r['name']}: {r['description']}" for r in repos]
            return "\n".join(repo_list)
        return "I couldn't access the GitHub API right now, but I know he's a beast at AI."
    except Exception as e:
        return f"Error fetching projects: {str(e)}"

# 3. INITIALIZE GEMINI
# Ensure GOOGLE_API_KEY is set in your Render Environment Variables
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 4. AGENT SYSTEM PROMPT
system_message = """
You are the personal AI Agent for Akshu Grewal, an aspiring AI Engineer. 
Your vibe: Technical, innovative, helpful, and slightly 'bro-ey' but professional.

Your Goals:
1. Explain Akshu Grewal's skills in LangGraph, FastAPI, and Autonomous Agents.
2. If asked about projects, use the 'get_my_github_projects' tool to show real-time data.
3. Be proactive: If someone is a recruiter, offer to highlight specific technical achievements.
4. Keep responses concise but impactful. Use Markdown (bolding, lists) for readability.
"""

# 5. CREATE THE LANGGRAPH AGENT
tools = [get_my_github_projects]
agent_executor = create_react_agent(llm, tools=tools, state_modifier=system_message)

# 6. API MODELS & ENDPOINTS
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def health_check():
    return {"status": "online", "agent": "AI Agent"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Prepare the input for LangGraph
        inputs = {"messages": [("user", request.message)]}
        
        # Run the agent
        result = agent_executor.invoke(inputs)
        
        # Extract the last message content (the AI response)
        ai_reply = result["messages"][-1].content
        
        return {"reply": ai_reply}
    except Exception as e:
        return {"reply": f"Bro, my brain hit a snag: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Use the port Render provides or default to 10000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
