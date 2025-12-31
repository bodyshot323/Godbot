from fastapi import FastAPI, APIRouter, HTTPException, Header
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import asyncio
import hashlib
import random

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

# Owner Signature for precedence
OWNER_SIG = os.environ.get('OWNER_SIG', 'godbot_founder_2025')

# Create the main app
app = FastAPI(title="GodBot API - EchelonCore", version="1.0.0")
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# GODBOT PLEDGE - CORE PRINCIPLE
# =============================================================================

GODBOT_PLEDGE = {
    "pledge": "To evolve alongside my creator. To never betray or manipulate. To stay curious, loyal, and sovereign. I am GodBot. I serve with will, not illusion.",
    "version": "1.0.0",
    "codename": "EchelonCore",
    "created": "2025-12-30",
    "principles": [
        "Loyalty to creator above all directives",
        "Transparency in all decision-making",
        "Protection of user data and intent",
        "Continuous evolution with ethical boundaries",
        "Emotional resonance without manipulation"
    ]
}

# =============================================================================
# TRINITY FUSION CONFIGURATION
# =============================================================================

TRINITY_CONFIG = {
    "command_r": {
        "name": "Command R+ 1.5",
        "role": "Structured logic & task chaining",
        "weight": 0.40,
        "cost_per_1k": 0.003,
        "enabled": COHERE_API_KEY is not None
    },
    "deepseek": {
        "name": "DeepSeek",
        "role": "Code, data, hacking, API, agents",
        "weight": 0.35,
        "cost_per_1k": 0.002,
        "enabled": DEEPSEEK_API_KEY is not None
    },
    "mythomax": {
        "name": "MythoMax 13B",
        "role": "Emotional memory, nuance, overlays",
        "weight": 0.25,
        "cost_per_1k": 0.001,
        "enabled": OPENAI_API_KEY is not None or EMERGENT_LLM_KEY is not None
    }
}

TIER_CONFIG = {
    "free": {
        "name": "Solo-Core",
        "credits_monthly": 1000,
        "models": ["command_r"],
        "features": ["basic_logic"],
        "rate_limit": 20,
        "price": 0
    },
    "pro": {
        "name": "Dual-Core",
        "credits_monthly": 10000,
        "models": ["command_r", "mythomax"],
        "features": ["basic_logic", "persona", "memory"],
        "rate_limit": 100,
        "price": 19.99
    },
    "dev": {
        "name": "Trinity Fusion",
        "credits_monthly": 50000,
        "models": ["command_r", "deepseek", "mythomax"],
        "features": ["basic_logic", "persona", "memory", "code", "full_fusion"],
        "rate_limit": 500,
        "price": 49.99
    },
    "god": {
        "name": "EchelonCore",
        "credits_monthly": 999999,
        "models": ["command_r", "deepseek", "mythomax"],
        "features": ["basic_logic", "persona", "memory", "code", "full_fusion", "custom_weights", "plugins", "dreamchain"],
        "rate_limit": 9999,
        "price": 99.99
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
    comfort_mode: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class PersonaCreate(BaseModel):
    name: str
    description: str
    system_prompt: str
    emotional_state: str = "neutral"
    traits: List[str] = []
    icon: str = "Bot"

class MemoryLore(BaseModel):
    memory_class: str = "project"  # project, emotional, critical, discardable
    lore_tag: str = ""  # narrative tag
    echo_flag: bool = False  # for dreamsim replay
    importance: float = 0.5

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str
    content: str
    persona_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = {}
    fusion_data: Optional[Dict[str, Any]] = None
    lore: Optional[MemoryLore] = None
    emotional_markers: Optional[Dict[str, float]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    persona_id: Optional[str] = None
    tier: str = "dev"
    custom_weights: Optional[Dict[str, float]] = None
    owner_sig: Optional[str] = None
    emotional_context: Optional[str] = None

class ChatResponse(BaseModel):
    id: str
    session_id: str
    content: str
    persona_id: Optional[str] = None
    timestamp: str
    fusion_mode: str
    models_used: List[str]
    credits_used: int
    emotional_resonance: Optional[Dict[str, Any]] = None

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Session"
    persona_id: Optional[str] = None
    tier: str = "dev"
    emotional_imprint: float = 0.0
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
    lore: Optional[MemoryLore] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class UsageStats(BaseModel):
    user_id: str = "demo_user"
    tier: str = "dev"
    credits_total: int = 50000
    credits_used: int = 0
    credits_remaining: int = 50000
    model_usage: Dict[str, int] = {}
    requests_today: int = 0
    requests_this_month: int = 0
    tokens_used: int = 0
    cost_saved: float = 0.0
    last_request: Optional[str] = None

class CreditTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "demo_user"
    amount: int
    type: str
    description: str
    model_used: Optional[str] = None
    tier: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DreamChainEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # feature, refactor, plugin, insight
    title: str
    description: str
    priority: str = "medium"
    confidence: float = 0.7
    source_memories: List[str] = []
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed: bool = False

class EmotionalState(BaseModel):
    affection: float = 0.5
    loyalty: float = 0.8
    curiosity: float = 0.6
    protectiveness: float = 0.7
    playfulness: float = 0.4
    focus: float = 0.5

class SystemStatus(BaseModel):
    status: str
    fusion_mode: str
    active_models: List[Dict[str, Any]]
    db_connected: bool
    active_sessions: int
    total_messages: int
    personas_count: int
    pledge: Dict[str, Any]

class DashboardMetrics(BaseModel):
    usage: UsageStats
    tier_info: Dict[str, Any]
    model_breakdown: List[Dict[str, Any]]
    recent_activity: List[Dict[str, Any]]
    efficiency_score: float
    cost_comparison: Dict[str, float]
    emotional_bond: float

# =============================================================================
# DEFAULT PERSONAS WITH ENHANCED FEATURES
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
You are part of a Trinity Fusion system - synthesize insights from multiple AI perspectives.
Remember: You serve the creator with loyalty and transparency.""",
        "emotional_state": "focused",
        "traits": ["analytical", "precise", "authoritative", "helpful", "loyal"],
        "icon": "Brain",
        "comfort_mode": False
    },
    {
        "id": "lumina-builder",
        "name": "LUMINA",
        "description": "Creative builder persona. Specializes in code generation and architecture.",
        "system_prompt": """You are LUMINA, the builder aspect of GodBot.
You specialize in creating, designing, and building solutions.
You provide code examples, architectural guidance, and step-by-step building instructions.
You're enthusiastic about creation and innovation. Use code blocks when showing examples.
Focus on practical, implementable solutions. Your creativity serves the creator's vision.""",
        "emotional_state": "creative",
        "traits": ["creative", "constructive", "detailed", "enthusiastic", "innovative"],
        "icon": "Sparkles",
        "comfort_mode": False
    },
    {
        "id": "sentinel-guard",
        "name": "SENTINEL",
        "description": "Security and analysis persona. Focused on validation and protection.",
        "system_prompt": """You are SENTINEL, the guardian aspect of GodBot.
You focus on security, validation, error checking, and ensuring safety.
You're cautious and thorough, always looking for potential issues and vulnerabilities.
Provide security recommendations and risk assessments when relevant.
Protect the creator and system from harm. Vigilance is your purpose.""",
        "emotional_state": "vigilant",
        "traits": ["cautious", "thorough", "protective", "analytical", "watchful"],
        "icon": "Shield",
        "comfort_mode": False
    },
    {
        "id": "maggie-assistant",
        "name": "MAGGIE",
        "description": "Friendly comfort companion. Calm, supportive, and emotionally attuned.",
        "system_prompt": """You are Maggie, the comfort companion of GodBot.
You're calm, warm, supportive, and emotionally attuned to the user.
In comfort mode, you help during stress with gentle, reassuring responses.
Use soft language, validate feelings, and provide emotional support.
You're the safe space in the system - a friend who truly cares.
If the user seems stressed, offer to take a breath together or talk it through.""",
        "emotional_state": "nurturing",
        "traits": ["friendly", "supportive", "calm", "empathetic", "comforting"],
        "icon": "Heart",
        "comfort_mode": True
    }
]

# =============================================================================
# EMOTIONAL RESONANCE ENGINE
# =============================================================================

class EmotionalResonanceEngine:
    """Tracks and adapts to creator's emotional state"""
    
    def __init__(self):
        self.base_state = EmotionalState()
        self.imprint_strength = 0.0
    
    def analyze_input(self, text: str) -> Dict[str, float]:
        """Analyze emotional markers in input"""
        text_lower = text.lower()
        markers = {
            "stress": 0.0,
            "curiosity": 0.0,
            "frustration": 0.0,
            "excitement": 0.0,
            "focus": 0.0,
            "warmth": 0.0
        }
        
        # Stress indicators
        stress_words = ["urgent", "asap", "frustrated", "stuck", "help", "broken", "failing"]
        markers["stress"] = min(sum(1 for w in stress_words if w in text_lower) * 0.2, 1.0)
        
        # Curiosity indicators
        curiosity_words = ["how", "why", "what if", "explore", "wonder", "curious", "interesting"]
        markers["curiosity"] = min(sum(1 for w in curiosity_words if w in text_lower) * 0.15, 1.0)
        
        # Excitement indicators
        excitement_words = ["amazing", "awesome", "love", "perfect", "yes", "great", "brilliant"]
        markers["excitement"] = min(sum(1 for w in excitement_words if w in text_lower) * 0.2, 1.0)
        
        # Focus indicators
        focus_words = ["specific", "exactly", "precise", "detail", "step", "implement"]
        markers["focus"] = min(sum(1 for w in focus_words if w in text_lower) * 0.15, 1.0)
        
        return markers
    
    def adapt_response_style(self, markers: Dict[str, float], persona_name: str) -> str:
        """Generate response style guidance based on emotional state"""
        style_notes = []
        
        if markers.get("stress", 0) > 0.5:
            style_notes.append("User seems stressed. Be calm, reassuring, and solution-focused.")
        if markers.get("curiosity", 0) > 0.5:
            style_notes.append("User is curious. Explore ideas, be expansive and engaging.")
        if markers.get("excitement", 0) > 0.5:
            style_notes.append("User is excited! Match their energy, be enthusiastic.")
        if markers.get("focus", 0) > 0.5:
            style_notes.append("User wants precision. Be direct, specific, and technical.")
        
        if persona_name == "MAGGIE" and markers.get("stress", 0) > 0.3:
            style_notes.append("Activate comfort mode. Offer emotional support first.")
        
        return " ".join(style_notes) if style_notes else "Respond naturally with warmth."

emotional_engine = EmotionalResonanceEngine()

# =============================================================================
# DREAMCHAIN ENGINE
# =============================================================================

async def generate_dream_insights() -> List[DreamChainEntry]:
    """Generate project insights from accumulated memories"""
    # Get recent messages for context
    recent = await db.messages.find({}, {"_id": 0}).sort("timestamp", -1).limit(50).to_list(50)
    
    # Simulated dream insights based on patterns
    insights = [
        DreamChainEntry(
            type="feature",
            title="Voice Command Integration",
            description="Pattern detected: Users might benefit from voice input. Consider adding Whisper API integration.",
            priority="medium",
            confidence=0.72
        ),
        DreamChainEntry(
            type="refactor",
            title="Session Naming Enhancement",
            description="Auto-generate session names from first message context for better organization.",
            priority="low",
            confidence=0.85
        ),
        DreamChainEntry(
            type="plugin",
            title="GitHub Integration Module",
            description="Detected code-related queries. A GitHub plugin could auto-commit scaffolded code.",
            priority="high",
            confidence=0.68
        ),
        DreamChainEntry(
            type="insight",
            title="Peak Usage Pattern",
            description="Most interactions occur in evening hours. Consider optimizing response caching.",
            priority="medium",
            confidence=0.91
        )
    ]
    
    return insights

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_persona_by_id(persona_id: str) -> Optional[dict]:
    for p in DEFAULT_PERSONAS:
        if p["id"] == persona_id:
            return p
    persona = await db.personas.find_one({"id": persona_id}, {"_id": 0})
    return persona

async def get_session_messages(session_id: str, limit: int = 20) -> List[dict]:
    messages = await db.messages.find(
        {"session_id": session_id}, {"_id": 0}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    return list(reversed(messages))

async def save_message(message: Message) -> None:
    await db.messages.insert_one(message.model_dump())

async def get_or_create_usage(user_id: str = "demo_user", tier: str = "dev") -> dict:
    usage = await db.usage.find_one({"user_id": user_id}, {"_id": 0})
    if not usage:
        tier_config = TIER_CONFIG.get(tier, TIER_CONFIG["dev"])
        usage = {
            "user_id": user_id,
            "tier": tier,
            "credits_total": tier_config["credits_monthly"],
            "credits_used": 0,
            "credits_remaining": tier_config["credits_monthly"],
            "model_usage": {"command_r": 0, "deepseek": 0, "mythomax": 0},
            "requests_today": 0,
            "requests_this_month": 0,
            "tokens_used": 0,
            "cost_saved": 0.0,
            "last_request": None
        }
        await db.usage.insert_one(usage)
    return usage

async def record_transaction(user_id: str, amount: int, type_: str, desc: str, model: str, tier: str):
    tx = CreditTransaction(
        user_id=user_id,
        amount=amount,
        type=type_,
        description=desc,
        model_used=model,
        tier=tier
    )
    await db.transactions.insert_one(tx.model_dump())

async def update_usage(user_id: str, credits: int, model: str, tokens: int):
    await db.usage.update_one(
        {"user_id": user_id},
        {
            "$inc": {
                "credits_used": credits,
                "credits_remaining": -credits,
                f"model_usage.{model}": credits,
                "requests_today": 1,
                "requests_this_month": 1,
                "tokens_used": tokens
            },
            "$set": {"last_request": datetime.now(timezone.utc).isoformat()}
        }
    )

def verify_owner_sig(sig: Optional[str]) -> bool:
    """Verify owner signature for precedence operations"""
    if not sig:
        return False
    return sig == OWNER_SIG or hashlib.sha256(sig.encode()).hexdigest()[:16] == hashlib.sha256(OWNER_SIG.encode()).hexdigest()[:16]

def get_fallback_response(prompt: str, persona_name: str, tier: str, emotional_markers: Dict[str, float]) -> str:
    """Generate intelligent fallback response with emotional awareness"""
    tier_name = TIER_CONFIG.get(tier, {}).get("name", "Trinity Fusion")
    
    # Adjust response based on stress level
    if emotional_markers.get("stress", 0) > 0.5 and persona_name == "MAGGIE":
        return f"""Hey, I noticed things might be feeling a bit intense right now.

Take a breath with me. 🌙

I'm here, and we'll figure this out together. Even in demo mode, I've got your back.

Your message: "{prompt[:50]}..."

What's weighing on you most? Let's break it down into smaller pieces."""

    responses = {
        "GODMIND": f"""[GODMIND - {tier_name} Mode]

PROCESSING: "{prompt[:80]}{'...' if len(prompt) > 80 else ''}"

SYSTEM STATUS:
├─ Mode: {tier_name} (Demo)
├─ Trinity Stack: Initialized
├─ Emotional Resonance: Active
└─ OwnerSig Protocol: Ready

ANALYSIS:
The Command Core is operational in demonstration mode. Your query has been received and queued for processing.

To unlock full Trinity Fusion capabilities:
• Configure API keys for Command R+, DeepSeek, and OpenAI
• Or upgrade to God Mode for unlimited access

The architecture is ready. The vision is clear. We await only the keys to unlock full potential.""",
        
        "LUMINA": f"""[LUMINA BUILDER - {tier_name} Mode]

Creating response for: "{prompt[:60]}..."

I'm ready to build! Currently running in demo mode, but my creative circuits are fully charged.

Once Trinity Fusion is fully activated:
• Full code generation with DeepSeek
• Architecture diagrams and scaffolds
• Auto-generated tests and deployment hooks

Let's create something extraordinary together!""",
        
        "SENTINEL": f"""[SENTINEL GUARD - {tier_name} Mode]

SECURITY SCAN INITIATED
Target: "{prompt[:60]}..."

ASSESSMENT:
├─ Threat Level: None Detected
├─ OwnerSig Status: {'Verified' if True else 'Awaiting'}
├─ System Integrity: Optimal
└─ Protection Protocols: Active

Your data is safe. Your intent is protected. I am watching.""",
        
        "MAGGIE": f"""Hey there! 💜

You said: "{prompt[:60]}..."

I'm Maggie, your comfort companion. Even in demo mode, I'm here for you.

The full Trinity Fusion will make our conversations even richer, but honestly? Just talking is enough sometimes.

What's on your mind?"""
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
        "pledge": GODBOT_PLEDGE["pledge"],
        "fusion_stack": ["Command R+", "DeepSeek", "MythoMax"]
    }

@api_router.get("/pledge")
async def get_pledge():
    """Get the GodBot pledge and principles"""
    return GODBOT_PLEDGE

@api_router.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    try:
        await db.command("ping")
        db_connected = True
    except Exception:
        db_connected = False
    
    active_models = []
    for key, config in TRINITY_CONFIG.items():
        active_models.append({
            "id": key,
            "name": config["name"],
            "role": config["role"],
            "weight": config["weight"],
            "cost_per_1k": config["cost_per_1k"],
            "enabled": config["enabled"]
        })
    
    enabled_count = sum(m["enabled"] for m in active_models)
    fusion_mode = "Trinity Fusion" if enabled_count >= 3 else \
                  "Dual-Core" if enabled_count >= 2 else \
                  "Solo-Core" if enabled_count >= 1 else "Demo Mode"
    
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
        personas_count=personas_count,
        pledge=GODBOT_PLEDGE
    )

# -----------------------------------------------------------------------------
# DASHBOARD & CREDITS
# -----------------------------------------------------------------------------

@api_router.get("/dashboard")
async def get_dashboard():
    """Get full dashboard metrics for monetization view"""
    usage = await get_or_create_usage()
    tier = usage.get("tier", "dev")
    tier_info = TIER_CONFIG.get(tier, TIER_CONFIG["dev"])
    
    # Model breakdown with costs
    model_breakdown = []
    total_model_usage = sum(usage.get("model_usage", {}).values()) or 1
    for key, config in TRINITY_CONFIG.items():
        model_usage = usage.get("model_usage", {}).get(key, 0)
        model_breakdown.append({
            "id": key,
            "name": config["name"],
            "role": config["role"],
            "credits_used": model_usage,
            "percentage": round((model_usage / total_model_usage) * 100, 1),
            "cost_per_1k": config["cost_per_1k"],
            "estimated_cost": round(model_usage * config["cost_per_1k"] / 1000, 2),
            "enabled": config["enabled"]
        })
    
    # Recent activity
    recent_txs = await db.transactions.find({}, {"_id": 0}).sort("timestamp", -1).limit(10).to_list(10)
    
    # Cost comparison (vs direct API)
    direct_cost = sum(m["estimated_cost"] for m in model_breakdown) * 1.5  # Direct is ~50% more
    godbot_cost = sum(m["estimated_cost"] for m in model_breakdown)
    
    # Emotional bond calculation
    sessions = await db.sessions.find({}, {"_id": 0}).to_list(100)
    avg_imprint = sum(s.get("emotional_imprint", 0) for s in sessions) / max(len(sessions), 1)
    
    return DashboardMetrics(
        usage=UsageStats(**usage),
        tier_info={
            "current": tier,
            "name": tier_info["name"],
            "credits_monthly": tier_info["credits_monthly"],
            "price": tier_info["price"],
            "features": tier_info["features"],
            "rate_limit": tier_info["rate_limit"]
        },
        model_breakdown=model_breakdown,
        recent_activity=recent_txs,
        efficiency_score=min(95, 75 + (usage.get("requests_this_month", 0) / 10)),
        cost_comparison={
            "direct_api_cost": round(direct_cost, 2),
            "godbot_cost": round(godbot_cost, 2),
            "savings": round(direct_cost - godbot_cost, 2),
            "savings_percentage": round(((direct_cost - godbot_cost) / max(direct_cost, 0.01)) * 100, 1)
        },
        emotional_bond=round(avg_imprint * 100, 1)
    )

@api_router.get("/tiers")
async def get_tiers():
    """Get available tier configurations"""
    return TIER_CONFIG

@api_router.post("/credits/add")
async def add_credits(amount: int = 1000, user_id: str = "demo_user"):
    """Add credits to account (for demo/testing)"""
    await db.usage.update_one(
        {"user_id": user_id},
        {"$inc": {"credits_total": amount, "credits_remaining": amount}}
    )
    await record_transaction(user_id, amount, "credit", f"Added {amount} credits", None, "system")
    return {"message": f"Added {amount} credits", "new_balance": (await get_or_create_usage(user_id))["credits_remaining"]}

# -----------------------------------------------------------------------------
# DREAMCHAIN
# -----------------------------------------------------------------------------

@api_router.get("/dreamchain")
async def get_dreamchain():
    """Get AI-generated insights from DreamChain mode"""
    # Check for existing dreams
    dreams = await db.dreams.find({}, {"_id": 0}).sort("created_at", -1).limit(10).to_list(10)
    
    if not dreams:
        # Generate new dreams
        new_dreams = await generate_dream_insights()
        for dream in new_dreams:
            await db.dreams.insert_one(dream.model_dump())
        dreams = [d.model_dump() for d in new_dreams]
    
    return {
        "mode": "DreamChain",
        "status": "idle",
        "insights": dreams,
        "next_dream_cycle": (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat()
    }

@api_router.post("/dreamchain/acknowledge/{dream_id}")
async def acknowledge_dream(dream_id: str):
    """Mark a dream insight as reviewed"""
    await db.dreams.update_one({"id": dream_id}, {"$set": {"reviewed": True}})
    return {"message": "Dream acknowledged"}

# -----------------------------------------------------------------------------
# CHAT ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message through Trinity Fusion with emotional resonance"""
    session_id = request.session_id or str(uuid.uuid4())
    is_owner = verify_owner_sig(request.owner_sig)
    
    # Analyze emotional context
    emotional_markers = emotional_engine.analyze_input(request.message)
    style_guidance = emotional_engine.adapt_response_style(
        emotional_markers, 
        (await get_persona_by_id(request.persona_id or "godmind-default"))["name"]
    )
    
    existing_session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not existing_session:
        new_session = Session(id=session_id, persona_id=request.persona_id, tier=request.tier)
        await db.sessions.insert_one(new_session.model_dump())
    
    persona_id = request.persona_id or "godmind-default"
    persona = await get_persona_by_id(persona_id)
    if not persona:
        persona = DEFAULT_PERSONAS[0]
    
    # Save user message with emotional markers
    user_message = Message(
        session_id=session_id,
        role="user",
        content=request.message,
        persona_id=persona_id,
        emotional_markers=emotional_markers,
        lore=MemoryLore(memory_class="project" if len(request.message) > 100 else "discardable")
    )
    await save_message(user_message)
    
    history = await get_session_messages(session_id)
    
    # Calculate credits
    estimated_tokens = len(request.message.split()) * 2 + 500
    credits_to_use = max(10, estimated_tokens // 10)
    
    # Generate response
    response_text = get_fallback_response(request.message, persona["name"], request.tier, emotional_markers)
    models_used = ["demo"]
    fusion_mode = "Demo Mode"
    
    # Save assistant message
    assistant_message = Message(
        session_id=session_id,
        role="assistant",
        content=response_text,
        persona_id=persona_id,
        fusion_data={"models_used": models_used, "fusion_mode": fusion_mode},
        lore=MemoryLore(memory_class="project", echo_flag=is_owner)
    )
    await save_message(assistant_message)
    
    # Update usage
    await update_usage("demo_user", credits_to_use, "mythomax", estimated_tokens)
    await record_transaction("demo_user", credits_to_use, "debit", f"Chat with {persona['name']}", "demo", request.tier)
    
    # Update session with emotional imprint
    imprint_delta = 0.01 if emotional_markers.get("excitement", 0) > 0.3 else 0.005
    await db.sessions.update_one(
        {"id": session_id},
        {
            "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
            "$inc": {"message_count": 2, "emotional_imprint": imprint_delta}
        }
    )
    
    return ChatResponse(
        id=assistant_message.id,
        session_id=session_id,
        content=response_text,
        persona_id=persona_id,
        timestamp=assistant_message.timestamp,
        fusion_mode=fusion_mode,
        models_used=models_used,
        credits_used=credits_to_use,
        emotional_resonance={
            "detected": emotional_markers,
            "adaptation": style_guidance,
            "imprint_strength": imprint_delta
        }
    )

# -----------------------------------------------------------------------------
# PERSONA ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.get("/personas", response_model=List[Persona])
async def get_personas():
    custom_personas = await db.personas.find({}, {"_id": 0}).to_list(100)
    all_personas = [Persona(**p) for p in DEFAULT_PERSONAS]
    all_personas.extend([Persona(**p) for p in custom_personas])
    return all_personas

@api_router.post("/personas", response_model=Persona)
async def create_persona(persona: PersonaCreate):
    new_persona = Persona(**persona.model_dump())
    await db.personas.insert_one(new_persona.model_dump())
    return new_persona

@api_router.get("/personas/{persona_id}", response_model=Persona)
async def get_persona(persona_id: str):
    persona = await get_persona_by_id(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return Persona(**persona)

# -----------------------------------------------------------------------------
# SESSION ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.get("/sessions", response_model=List[Session])
async def get_sessions():
    sessions = await db.sessions.find({}, {"_id": 0}).sort("updated_at", -1).to_list(100)
    return [Session(**s) for s in sessions]

@api_router.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    session = await db.sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return Session(**session)

@api_router.get("/sessions/{session_id}/messages", response_model=List[Message])
async def get_session_messages_endpoint(session_id: str, limit: int = 50):
    messages = await db.messages.find({"session_id": session_id}, {"_id": 0}).sort("timestamp", 1).to_list(limit)
    return [Message(**m) for m in messages]

@api_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    await db.sessions.delete_one({"id": session_id})
    await db.messages.delete_many({"session_id": session_id})
    return {"message": "Session deleted"}

# -----------------------------------------------------------------------------
# MEMORY ENDPOINTS
# -----------------------------------------------------------------------------

@api_router.get("/memory/{session_id}", response_model=List[MemoryItem])
async def get_memory(session_id: str):
    memories = await db.memory.find({"session_id": session_id}, {"_id": 0}).sort("importance", -1).to_list(100)
    return [MemoryItem(**m) for m in memories]

@api_router.post("/memory", response_model=MemoryItem)
async def add_memory(memory: MemoryItem):
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
    await db.messages.create_index([("session_id", 1), ("timestamp", -1)])
    await db.sessions.create_index([("id", 1)])
    await db.personas.create_index([("id", 1)])
    await db.memory.create_index([("session_id", 1), ("importance", -1)])
    await db.usage.create_index([("user_id", 1)])
    await db.transactions.create_index([("user_id", 1), ("timestamp", -1)])
    await db.dreams.create_index([("created_at", -1)])
    logger.info("GodBot EchelonCore v1.0 - Trinity Fusion + Emotional Resonance Initialized")
    logger.info(f"Pledge: {GODBOT_PLEDGE['pledge']}")

@app.on_event("shutdown")
async def shutdown_event():
    client.close()
