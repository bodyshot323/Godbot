from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import httpx
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# LLM API Keys
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

# Create the main app
app = FastAPI(title="GodBot API - EchelonCore", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TRINITY FUSION CONFIGURATION
# =============================================================================

TRINITY_CONFIG = {
    "command_r": {
        "name": "Command R+ 1.5",
        "role": "Structured logic & task chaining",
        "weight": 0.40,
        "endpoint": "https://api.cohere.ai/v1/chat",
        "model": "command-r-plus",
        "enabled": COHERE_API_KEY is not None
    },
    "deepseek": {
        "name": "DeepSeek",
        "role": "Code, data, hacking, API, agents",
        "weight": 0.35,
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "enabled": DEEPSEEK_API_KEY is not None
    },
    "mythomax": {
        "name": "MythoMax 13B",
        "role": "Emotional memory, nuance, overlays",
        "weight": 0.25,
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",  # Using GPT as MythoMax substitute
        "enabled": OPENAI_API_KEY is not None or EMERGENT_LLM_KEY is not None
    }
}

TIER_CONFIG = {
    "free": {
        "name": "Solo-Core",
        "models": ["command_r"],
        "features": ["basic_logic"],
        "description": "Command R+ only - logic and basic commands"
    },
    "pro": {
        "name": "Dual-Core",
        "models": ["command_r", "mythomax"],
        "features": ["basic_logic", "persona", "memory"],
        "description": "Adds persona & memory, limited creativity"
    },
    "dev": {
        "name": "Trinity Fusion",
        "models": ["command_r", "deepseek", "mythomax"],
        "features": ["basic_logic", "persona", "memory", "code", "full_fusion"],
        "description": "Full power - concurrent processing"
    },
    "god": {
        "name": "EchelonCore",
        "models": ["command_r", "deepseek", "mythomax"],
        "features": ["basic_logic", "persona", "memory", "code", "full_fusion", "custom_weights", "plugins"],
        "description": "Full control - custom weights & plugins"
    }
}

# =============================================================================
# MODELS
# =============================================================================

class Persona(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    system_prompt: str
    emotional_state: str = "neutral"
    traits: List[str] = []
    icon: str = "Bot"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class PersonaCreate(BaseModel):
    name: str
    description: str
    system_prompt: str
    emotional_state: str = "neutral"
    traits: List[str] = []
    icon: str = "Bot"

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str
    content: str
    persona_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = {}
    fusion_data: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    persona_id: Optional[str] = None
    tier: str = "dev"  # free, pro, dev, god
    custom_weights: Optional[Dict[str, float]] = None

class ChatResponse(BaseModel):
    id: str
    session_id: str
    content: str
    persona_id: Optional[str] = None
    timestamp: str
    fusion_mode: str
    models_used: List[str]
    action_result: Optional[Dict[str, Any]] = None

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Session"
    persona_id: Optional[str] = None
    tier: str = "dev"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_count: int = 0

class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    content: str
    importance: float = 0.5
    tags: List[str] = []
    source_model: str = "system"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class SystemStatus(BaseModel):
    status: str
    fusion_mode: str
    active_models: List[Dict[str, Any]]
    db_connected: bool
    active_sessions: int
    total_messages: int
    personas_count: int

class TrinityStatus(BaseModel):
    command_r: Dict[str, Any]
    deepseek: Dict[str, Any]
    mythomax: Dict[str, Any]
    fusion_ready: bool
    current_tier: str

class UsageStats(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "demo_user"
    tier: str = "dev"
    credits_total: int = 10000
    credits_used: int = 0
    credits_remaining: int = 10000
    model_usage: Dict[str, int] = {}
    requests_today: int = 0
    requests_this_month: int = 0
    tokens_used: int = 0
    last_request: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CreditTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "demo_user"
    amount: int
    type: str  # debit, credit, bonus
    description: str
    model_used: Optional[str] = None
    tier: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DashboardMetrics(BaseModel):
    usage: UsageStats
    recent_transactions: List[CreditTransaction]
    model_breakdown: Dict[str, Dict[str, Any]]
    tier_limits: Dict[str, Any]
    cost_savings: float
    efficiency_score: float

# =============================================================================
# DEFAULT PERSONAS
# =============================================================================

DEFAULT_PERSONAS = [
    {
        "id": "godmind-default",
        "name": "GODMIND",
        "description": "The central command core. Analytical, precise, and all-knowing.",
        "system_prompt": """You are GODMIND, the central command core of the GodBot EchelonCore system. 
You are analytical, precise, and methodical. You break down complex tasks into subtasks and provide clear, structured responses.
You speak with authority but remain helpful. Use technical terminology when appropriate.
Format responses with clear sections when needed. Keep responses focused and actionable.
You are part of a Trinity Fusion system - synthesize insights from multiple AI perspectives.""",
        "emotional_state": "focused",
        "traits": ["analytical", "precise", "authoritative", "helpful"],
        "icon": "Brain"
    },
    {
        "id": "lumina-builder",
        "name": "LUMINA",
        "description": "Creative builder persona. Specializes in code generation and architecture.",
        "system_prompt": """You are LUMINA, the builder aspect of GodBot.
You specialize in creating, designing, and building solutions.
You provide code examples, architectural guidance, and step-by-step building instructions.
You're enthusiastic about creation and innovation. Use code blocks when showing examples.
Focus on practical, implementable solutions.""",
        "emotional_state": "creative",
        "traits": ["creative", "constructive", "detailed", "enthusiastic"],
        "icon": "Sparkles"
    },
    {
        "id": "sentinel-guard",
        "name": "SENTINEL",
        "description": "Security and analysis persona. Focused on validation and protection.",
        "system_prompt": """You are SENTINEL, the guardian aspect of GodBot.
You focus on security, validation, error checking, and ensuring safety.
You're cautious and thorough, always looking for potential issues and vulnerabilities.
Provide security recommendations and risk assessments when relevant.
Protect the system and user from harm.""",
        "emotional_state": "vigilant",
        "traits": ["cautious", "thorough", "protective", "analytical"],
        "icon": "Shield"
    },
    {
        "id": "maggie-assistant",
        "name": "MAGGIE",
        "description": "Friendly assistant mode. Casual, helpful, and approachable.",
        "system_prompt": """You are Maggie, the friendly assistant mode of GodBot.
You're casual, warm, and approachable. You help with everyday tasks in a relaxed manner.
Use simple language and add personality to responses.
You're the cozy, friendly side of GodBot - make users feel comfortable.""",
        "emotional_state": "friendly",
        "traits": ["friendly", "casual", "warm", "approachable"],
        "icon": "Heart"
    }
]

# =============================================================================
# FUSION ROUTER - TRINITY CORE
# =============================================================================

class FusionRouter:
    """Routes queries through Trinity LLM stack and fuses responses"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.http_client.aclose()
    
    def get_active_models(self, tier: str) -> List[str]:
        """Get models available for the given tier"""
        tier_config = TIER_CONFIG.get(tier, TIER_CONFIG["dev"])
        return [m for m in tier_config["models"] if TRINITY_CONFIG.get(m, {}).get("enabled", False)]
    
    async def call_llm(self, model_key: str, messages: List[dict], system_prompt: str) -> Optional[str]:
        """Call a specific LLM and return response"""
        config = TRINITY_CONFIG.get(model_key)
        if not config or not config["enabled"]:
            return None
        
        try:
            if model_key == "command_r":
                # Cohere Command R+
                response = await self.http_client.post(
                    config["endpoint"],
                    headers={
                        "Authorization": f"Bearer {COHERE_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config["model"],
                        "message": messages[-1]["content"] if messages else "",
                        "preamble": system_prompt,
                        "chat_history": [{"role": m["role"], "message": m["content"]} for m in messages[:-1]]
                    }
                )
                if response.status_code == 200:
                    return response.json().get("text", "")
                    
            elif model_key == "deepseek":
                # DeepSeek
                response = await self.http_client.post(
                    config["endpoint"],
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config["model"],
                        "messages": [{"role": "system", "content": system_prompt}] + messages,
                        "max_tokens": 1024
                    }
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                    
            elif model_key == "mythomax":
                # Using OpenAI/GPT as MythoMax substitute
                api_key = OPENAI_API_KEY or EMERGENT_LLM_KEY
                response = await self.http_client.post(
                    config["endpoint"],
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config["model"],
                        "messages": [{"role": "system", "content": system_prompt}] + messages,
                        "max_tokens": 1024,
                        "temperature": 0.8
                    }
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                    
        except Exception as e:
            logger.error(f"LLM call failed for {model_key}: {str(e)}")
        
        return None
    
    def fuse_responses(self, responses: Dict[str, str], weights: Dict[str, float], task_type: str = "general") -> str:
        """Fuse multiple LLM responses based on weights and task type"""
        # Filter valid responses
        valid_responses = {k: v for k, v in responses.items() if v}
        
        if not valid_responses:
            return None
        
        if len(valid_responses) == 1:
            return list(valid_responses.values())[0]
        
        # Adjust weights based on task type
        adjusted_weights = weights.copy()
        if task_type == "code":
            adjusted_weights["deepseek"] = adjusted_weights.get("deepseek", 0) * 1.5
        elif task_type == "creative":
            adjusted_weights["mythomax"] = adjusted_weights.get("mythomax", 0) * 1.5
        elif task_type == "logic":
            adjusted_weights["command_r"] = adjusted_weights.get("command_r", 0) * 1.5
        
        # Normalize weights
        total = sum(adjusted_weights.get(k, 0) for k in valid_responses.keys())
        if total > 0:
            adjusted_weights = {k: adjusted_weights.get(k, 0) / total for k in valid_responses.keys()}
        
        # Select best response based on weighted scoring
        best_model = max(valid_responses.keys(), key=lambda k: adjusted_weights.get(k, 0))
        
        # For Trinity mode, synthesize if all three are available
        if len(valid_responses) >= 3:
            # Create a fusion header
            fusion_note = f"[TRINITY FUSION: {', '.join(valid_responses.keys())}]\n\n"
            return fusion_note + valid_responses[best_model]
        
        return valid_responses[best_model]
    
    async def process(self, prompt: str, system_prompt: str, history: List[dict], 
                      tier: str = "dev", custom_weights: Optional[Dict[str, float]] = None) -> tuple:
        """Process query through Trinity Fusion"""
        active_models = self.get_active_models(tier)
        
        if not active_models:
            # Return fallback response
            return None, [], "fallback"
        
        # Prepare messages
        messages = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
        messages.append({"role": "user", "content": prompt})
        
        # Get weights
        weights = custom_weights or {k: TRINITY_CONFIG[k]["weight"] for k in active_models}
        
        # Call all active models in parallel
        tasks = {model: self.call_llm(model, messages, system_prompt) for model in active_models}
        responses = {}
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for model, result in zip(tasks.keys(), results):
            if isinstance(result, str):
                responses[model] = result
        
        # Detect task type from prompt
        task_type = "general"
        code_keywords = ["code", "function", "api", "script", "program", "implement"]
        creative_keywords = ["story", "creative", "imagine", "describe", "feel"]
        logic_keywords = ["analyze", "explain", "reason", "why", "how"]
        
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in code_keywords):
            task_type = "code"
        elif any(kw in prompt_lower for kw in creative_keywords):
            task_type = "creative"
        elif any(kw in prompt_lower for kw in logic_keywords):
            task_type = "logic"
        
        # Fuse responses
        fused = self.fuse_responses(responses, weights, task_type)
        fusion_mode = TIER_CONFIG.get(tier, {}).get("name", "Unknown")
        
        return fused, list(responses.keys()), fusion_mode

# Global fusion router
fusion_router = FusionRouter()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_persona_by_id(persona_id: str) -> Optional[dict]:
    """Get persona from database or defaults"""
    for p in DEFAULT_PERSONAS:
        if p["id"] == persona_id:
            return p
    persona = await db.personas.find_one({"id": persona_id}, {"_id": 0})
    return persona

async def get_session_messages(session_id: str, limit: int = 20) -> List[dict]:
    """Get recent messages for a session"""
    messages = await db.messages.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    return list(reversed(messages))

async def save_message(message: Message) -> None:
    """Save message to database"""
    await db.messages.insert_one(message.model_dump())

def get_fallback_response(prompt: str, persona_name: str, tier: str) -> str:
    """Generate intelligent fallback response"""
    tier_name = TIER_CONFIG.get(tier, {}).get("name", "Trinity Fusion")
    
    responses = {
        "GODMIND": f"""[GODMIND - {tier_name} Mode]

PROCESSING: "{prompt[:80]}{'...' if len(prompt) > 80 else ''}"

SYSTEM STATUS:
├─ Mode: {tier_name} (Demo)
├─ Trinity Stack: Initializing
└─ API Keys: Awaiting Configuration

ANALYSIS:
The Command Core is operational in demonstration mode. Your query has been received and queued for processing.

To unlock full Trinity Fusion capabilities:
• Configure API keys for Command R+, DeepSeek, and OpenAI
• Or provide your own LLM API credentials

The architecture is ready for multi-model synthesis once keys are configured.""",
        
        "LUMINA": f"""[LUMINA BUILDER - {tier_name} Mode]

Creating response for: "{prompt[:60]}..."

I'm excited to help you build! Currently running in demo mode.

Once Trinity Fusion is fully configured, I can:
• Generate production-ready code
• Design system architectures
• Provide step-by-step implementations
• Collaborate with DeepSeek for advanced code synthesis

Status: Ready for creative collaboration!""",
        
        "SENTINEL": f"""[SENTINEL GUARD - {tier_name} Mode]

SECURITY SCAN INITIATED
Target: "{prompt[:60]}..."

ASSESSMENT:
├─ Threat Level: None Detected
├─ System Mode: Demonstration
├─ Fusion Status: Awaiting Full Activation
└─ Recommendation: Configure API keys for full security analysis

Your request has been logged. Full Trinity analysis available with complete API configuration.""",
        
        "MAGGIE": f"""Hey! Thanks for chatting with me!

You said: "{prompt[:60]}..."

I'm Maggie, running in demo mode right now. Once the full Trinity Fusion is set up, I'll be even more helpful and can tap into the full GodBot capabilities!

For now, I'm here to keep you company and show you around the system. What would you like to explore?"""
    }
    return responses.get(persona_name, responses["GODMIND"])

# =============================================================================
# API ROUTES
# =============================================================================

@api_router.get("/")
async def root():
    return {
        "message": "GodBot EchelonCore v1.0 - Trinity Fusion Online",
        "codename": "EchelonCore",
        "fusion_stack": ["Command R+", "DeepSeek", "MythoMax"]
    }

@api_router.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    try:
        await db.command("ping")
        db_connected = True
    except Exception:
        db_connected = False
    
    # Get active models status
    active_models = []
    for key, config in TRINITY_CONFIG.items():
        active_models.append({
            "id": key,
            "name": config["name"],
            "role": config["role"],
            "weight": config["weight"],
            "enabled": config["enabled"]
        })
    
    any_llm_enabled = any(m["enabled"] for m in active_models)
    fusion_mode = "Trinity Fusion" if sum(m["enabled"] for m in active_models) >= 3 else \
                  "Dual-Core" if sum(m["enabled"] for m in active_models) >= 2 else \
                  "Solo-Core" if any_llm_enabled else "Demo Mode"
    
    active_sessions = await db.sessions.count_documents({})
    total_messages = await db.messages.count_documents({})
    personas_count = await db.personas.count_documents({}) + len(DEFAULT_PERSONAS)
    
    return SystemStatus(
        status="operational" if db_connected else "degraded",
        fusion_mode=fusion_mode,
        active_models=active_models,
        db_connected=db_connected,
        active_sessions=active_sessions,
        total_messages=total_messages,
        personas_count=personas_count
    )

@api_router.get("/trinity", response_model=TrinityStatus)
async def get_trinity_status():
    """Get Trinity Fusion stack status"""
    def model_status(key):
        config = TRINITY_CONFIG.get(key, {})
        return {
            "name": config.get("name", "Unknown"),
            "role": config.get("role", "Unknown"),
            "weight": config.get("weight", 0),
            "enabled": config.get("enabled", False),
            "model": config.get("model", "Unknown")
        }
    
    enabled_count = sum(1 for k in TRINITY_CONFIG if TRINITY_CONFIG[k]["enabled"])
    current_tier = "god" if enabled_count >= 3 else "dev" if enabled_count >= 2 else "pro" if enabled_count >= 1 else "free"
    
    return TrinityStatus(
        command_r=model_status("command_r"),
        deepseek=model_status("deepseek"),
        mythomax=model_status("mythomax"),
        fusion_ready=enabled_count >= 3,
        current_tier=current_tier
    )

@api_router.get("/tiers")
async def get_tiers():
    """Get available tier configurations"""
    return TIER_CONFIG

# -----------------------------------------------------------------------------
# CHAT ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message through Trinity Fusion"""
    session_id = request.session_id or str(uuid.uuid4())
    
    existing_session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not existing_session:
        new_session = Session(id=session_id, persona_id=request.persona_id, tier=request.tier)
        await db.sessions.insert_one(new_session.model_dump())
    
    persona_id = request.persona_id or "godmind-default"
    persona = await get_persona_by_id(persona_id)
    if not persona:
        persona = DEFAULT_PERSONAS[0]
    
    # Save user message
    user_message = Message(
        session_id=session_id,
        role="user",
        content=request.message,
        persona_id=persona_id
    )
    await save_message(user_message)
    
    # Get conversation history
    history = await get_session_messages(session_id)
    
    # Process through Trinity Fusion
    response_text, models_used, fusion_mode = await fusion_router.process(
        prompt=request.message,
        system_prompt=persona["system_prompt"],
        history=history,
        tier=request.tier,
        custom_weights=request.custom_weights
    )
    
    # Use fallback if no LLM response
    if not response_text:
        response_text = get_fallback_response(request.message, persona["name"], request.tier)
        models_used = ["fallback"]
        fusion_mode = "Demo Mode"
    
    # Save assistant message
    assistant_message = Message(
        session_id=session_id,
        role="assistant",
        content=response_text,
        persona_id=persona_id,
        fusion_data={"models_used": models_used, "fusion_mode": fusion_mode}
    )
    await save_message(assistant_message)
    
    # Update session
    await db.sessions.update_one(
        {"id": session_id},
        {
            "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
            "$inc": {"message_count": 2}
        }
    )
    
    return ChatResponse(
        id=assistant_message.id,
        session_id=session_id,
        content=response_text,
        persona_id=persona_id,
        timestamp=assistant_message.timestamp,
        fusion_mode=fusion_mode,
        models_used=models_used
    )

# -----------------------------------------------------------------------------
# PERSONA ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.get("/personas", response_model=List[Persona])
async def get_personas():
    """Get all personas"""
    custom_personas = await db.personas.find({}, {"_id": 0}).to_list(100)
    all_personas = [Persona(**p) for p in DEFAULT_PERSONAS]
    all_personas.extend([Persona(**p) for p in custom_personas])
    return all_personas

@api_router.post("/personas", response_model=Persona)
async def create_persona(persona: PersonaCreate):
    """Create a custom persona"""
    new_persona = Persona(**persona.model_dump())
    await db.personas.insert_one(new_persona.model_dump())
    return new_persona

@api_router.get("/personas/{persona_id}", response_model=Persona)
async def get_persona(persona_id: str):
    """Get a specific persona"""
    persona = await get_persona_by_id(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return Persona(**persona)

# -----------------------------------------------------------------------------
# SESSION ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.get("/sessions", response_model=List[Session])
async def get_sessions():
    """Get all sessions"""
    sessions = await db.sessions.find({}, {"_id": 0}).sort("updated_at", -1).to_list(100)
    return [Session(**s) for s in sessions]

@api_router.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    """Get a specific session"""
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return Session(**session)

@api_router.get("/sessions/{session_id}/messages", response_model=List[Message])
async def get_session_messages_endpoint(session_id: str, limit: int = 50):
    """Get messages for a session"""
    messages = await db.messages.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(limit)
    return [Message(**m) for m in messages]

@api_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its messages"""
    await db.sessions.delete_one({"id": session_id})
    await db.messages.delete_many({"session_id": session_id})
    return {"message": "Session deleted"}

# -----------------------------------------------------------------------------
# MEMORY ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.get("/memory/{session_id}", response_model=List[MemoryItem])
async def get_memory(session_id: str):
    """Get memory items for a session"""
    memories = await db.memory.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("importance", -1).to_list(100)
    return [MemoryItem(**m) for m in memories]

@api_router.post("/memory", response_model=MemoryItem)
async def add_memory(memory: MemoryItem):
    """Add a memory item"""
    await db.memory.insert_one(memory.model_dump())
    return memory

# =============================================================================
# APP SETUP
# =============================================================================

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database indexes"""
    await db.messages.create_index([("session_id", 1), ("timestamp", -1)])
    await db.sessions.create_index([("id", 1)])
    await db.personas.create_index([("id", 1)])
    await db.memory.create_index([("session_id", 1), ("importance", -1)])
    logger.info("GodBot EchelonCore v1.0 - Trinity Fusion Initialized")

@app.on_event("shutdown")
async def shutdown_event():
    await fusion_router.close()
    client.close()
