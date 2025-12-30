import { useState, useEffect, useRef, useCallback } from "react";
import "@/App.css";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { 
  Brain, Sparkles, Shield, Heart, Send, Plus, Trash2, 
  Terminal, Cpu, Database, Zap, ChevronDown,
  MessageSquare, Bot, User, Menu, X, Layers, Activity,
  Lock, Unlock, Code, Settings2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Icon mapping for personas
const PERSONA_ICONS = {
  Brain: Brain,
  Sparkles: Sparkles,
  Shield: Shield,
  Heart: Heart,
  Bot: Bot,
};

// Tier badge colors
const TIER_COLORS = {
  free: "bg-muted text-muted-foreground",
  pro: "bg-secondary/20 text-secondary border-secondary/50",
  dev: "bg-primary/20 text-primary border-primary/50",
  god: "bg-gradient-to-r from-primary/20 to-secondary/20 text-white border-primary/50",
};

// Status indicator component
const StatusIndicator = ({ enabled, label }) => (
  <div className="flex items-center gap-2">
    <div className={`w-2 h-2 rounded-full ${
      enabled ? 'bg-accent animate-pulse' : 'bg-muted-foreground/30'
    }`} />
    <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
  </div>
);

// Trinity Model Status
const TrinityModelBadge = ({ model, enabled }) => (
  <div className={`flex items-center gap-2 px-3 py-1.5 border ${
    enabled ? 'border-accent/50 bg-accent/10' : 'border-border bg-muted/20'
  }`}>
    <div className={`w-1.5 h-1.5 rounded-full ${enabled ? 'bg-accent' : 'bg-muted-foreground/30'}`} />
    <span className={`text-xs font-bold uppercase tracking-wider ${enabled ? 'text-accent' : 'text-muted-foreground'}`}>
      {model}
    </span>
  </div>
);

// Message bubble component
const MessageBubble = ({ message, persona, fusionMode, modelsUsed }) => {
  const isUser = message.role === "user";
  const PersonaIcon = persona ? PERSONA_ICONS[persona.icon] || Bot : Bot;
  
  return (
    <div 
      data-testid={`message-${message.id}`}
      className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''} mb-6 animate-in fade-in slide-in-from-bottom-2 duration-300`}
    >
      <div className={`flex-shrink-0 w-10 h-10 rounded-none flex items-center justify-center border ${
        isUser 
          ? 'bg-secondary/20 border-secondary/50' 
          : 'bg-primary/20 border-primary/50 neon-glow'
      }`}>
        {isUser ? (
          <User className="w-5 h-5 text-secondary" />
        ) : (
          <PersonaIcon className="w-5 h-5 text-primary" />
        )}
      </div>
      <div className={`flex-1 ${isUser ? 'text-right' : ''}`}>
        <div className={`flex items-center gap-2 mb-1 ${isUser ? 'justify-end' : ''}`}>
          <span className={`text-xs uppercase tracking-wider ${
            isUser ? 'text-secondary' : 'text-primary'
          }`}>
            {isUser ? 'USER' : persona?.name || 'GODMIND'}
          </span>
          {!isUser && fusionMode && (
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-primary/30 text-primary/70">
              {fusionMode}
            </Badge>
          )}
          <span className="text-xs text-muted-foreground">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <div className={`p-4 border ${
          isUser 
            ? 'bg-secondary/10 border-secondary/30 text-foreground' 
            : 'glass border-primary/30'
        }`}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>
        {!isUser && modelsUsed && modelsUsed.length > 0 && modelsUsed[0] !== 'fallback' && (
          <div className="flex gap-2 mt-2">
            {modelsUsed.map((model) => (
              <span key={model} className="text-[10px] text-muted-foreground uppercase tracking-wider">
                {model}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Sidebar component
const Sidebar = ({ sessions, currentSession, onSelectSession, onNewSession, onDeleteSession, isOpen, onClose, trinityStatus }) => (
  <div className={`
    fixed md:relative inset-y-0 left-0 z-50
    w-72 border-r border-border bg-card/90 backdrop-blur-xl
    transform transition-transform duration-300 ease-out
    ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
  `}>
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary/20 border border-primary/50 flex items-center justify-center neon-glow">
              <Layers className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="font-heading text-lg font-bold text-primary neon-text">GODBOT</h1>
              <p className="text-xs text-muted-foreground">EchelonCore v1.0</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="md:hidden p-2 hover:bg-muted/50 transition-colors"
            data-testid="close-sidebar-btn"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>
      
      {/* Trinity Status Panel */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-primary" />
          <span className="text-xs font-bold uppercase tracking-wider text-primary">Trinity Fusion</span>
        </div>
        <div className="space-y-2">
          <TrinityModelBadge model="Command R+" enabled={trinityStatus?.command_r?.enabled} />
          <TrinityModelBadge model="DeepSeek" enabled={trinityStatus?.deepseek?.enabled} />
          <TrinityModelBadge model="MythoMax" enabled={trinityStatus?.mythomax?.enabled} />
        </div>
        <div className={`mt-3 p-2 border text-center ${
          trinityStatus?.fusion_ready ? 'border-accent/50 bg-accent/10' : 'border-muted bg-muted/20'
        }`}>
          <span className={`text-xs font-bold uppercase tracking-wider ${
            trinityStatus?.fusion_ready ? 'text-accent' : 'text-muted-foreground'
          }`}>
            {trinityStatus?.fusion_ready ? 'FUSION READY' : 'DEMO MODE'}
          </span>
        </div>
      </div>
      
      {/* New Session Button */}
      <div className="p-4">
        <Button 
          onClick={onNewSession}
          className="w-full rounded-none border border-primary bg-primary/10 text-primary hover:bg-primary hover:text-black transition-all duration-300 font-bold uppercase tracking-widest text-xs"
          data-testid="new-session-btn"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Session
        </Button>
      </div>
      
      {/* Sessions List */}
      <ScrollArea className="flex-1 px-4">
        <div className="space-y-2 pb-4">
          {sessions.map((session) => (
            <div
              key={session.id}
              data-testid={`session-${session.id}`}
              className={`group p-3 border cursor-pointer transition-all duration-300 ${
                currentSession?.id === session.id
                  ? 'border-primary/50 bg-primary/10'
                  : 'border-border hover:border-primary/30 hover:bg-muted/30'
              }`}
              onClick={() => onSelectSession(session)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm truncate">{session.name}</span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-destructive/20 transition-all"
                  data-testid={`delete-session-${session.id}`}
                >
                  <Trash2 className="w-3 h-3 text-destructive" />
                </button>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {session.message_count || 0} messages
              </p>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  </div>
);

// Main App Component
function App() {
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [personas, setPersonas] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [selectedTier, setSelectedTier] = useState("dev");
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [trinityStatus, setTrinityStatus] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Fetch initial data
  useEffect(() => {
    fetchPersonas();
    fetchSessions();
    fetchStatus();
    fetchTrinityStatus();
    const statusInterval = setInterval(() => {
      fetchStatus();
      fetchTrinityStatus();
    }, 30000);
    return () => clearInterval(statusInterval);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch system status
  const fetchStatus = async () => {
    try {
      const response = await axios.get(`${API}/status`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error("Failed to fetch status:", error);
    }
  };

  // Fetch Trinity status
  const fetchTrinityStatus = async () => {
    try {
      const response = await axios.get(`${API}/trinity`);
      setTrinityStatus(response.data);
    } catch (error) {
      console.error("Failed to fetch trinity status:", error);
    }
  };

  // Fetch personas
  const fetchPersonas = async () => {
    try {
      const response = await axios.get(`${API}/personas`);
      setPersonas(response.data);
      const godmind = response.data.find(p => p.id === "godmind-default");
      setSelectedPersona(godmind || response.data[0]);
    } catch (error) {
      console.error("Failed to fetch personas:", error);
      toast.error("Failed to load personas");
    }
  };

  // Fetch sessions
  const fetchSessions = async () => {
    try {
      const response = await axios.get(`${API}/sessions`);
      setSessions(response.data);
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    }
  };

  // Fetch messages for session
  const fetchMessages = useCallback(async (sessionId) => {
    try {
      const response = await axios.get(`${API}/sessions/${sessionId}/messages`);
      setMessages(response.data);
    } catch (error) {
      console.error("Failed to fetch messages:", error);
    }
  }, []);

  // Select session
  const handleSelectSession = useCallback((session) => {
    setCurrentSession(session);
    fetchMessages(session.id);
    setSidebarOpen(false);
  }, [fetchMessages]);

  // Create new session
  const handleNewSession = () => {
    setCurrentSession(null);
    setMessages([]);
    setSidebarOpen(false);
  };

  // Delete session
  const handleDeleteSession = async (sessionId) => {
    try {
      await axios.delete(`${API}/sessions/${sessionId}`);
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      if (currentSession?.id === sessionId) {
        setCurrentSession(null);
        setMessages([]);
      }
      toast.success("Session deleted");
    } catch (error) {
      toast.error("Failed to delete session");
    }
  };

  // Send message
  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await axios.post(`${API}/chat`, {
        message: input,
        session_id: currentSession?.id,
        persona_id: selectedPersona?.id,
        tier: selectedTier,
      });

      const assistantMessage = {
        id: response.data.id,
        role: "assistant",
        content: response.data.content,
        persona_id: response.data.persona_id,
        timestamp: response.data.timestamp,
        fusion_mode: response.data.fusion_mode,
        models_used: response.data.models_used,
      };

      setMessages(prev => [...prev, assistantMessage]);

      if (!currentSession) {
        setCurrentSession({ id: response.data.session_id, name: "New Session", message_count: 2 });
        fetchSessions();
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to send message");
      setMessages(prev => prev.filter(m => m.id !== userMessage.id));
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const PersonaIcon = selectedPersona ? PERSONA_ICONS[selectedPersona.icon] || Bot : Bot;

  return (
    <div className="h-screen flex bg-background text-foreground overflow-hidden">
      <Toaster 
        position="top-right" 
        toastOptions={{
          style: {
            background: '#0a0a0f',
            border: '1px solid rgba(217, 70, 239, 0.5)',
            color: '#e2e8f0',
          },
        }}
      />
      
      {/* Sidebar Overlay for Mobile */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <Sidebar 
        sessions={sessions}
        currentSession={currentSession}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        trinityStatus={trinityStatus}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center justify-between px-4 md:px-6">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-2 hover:bg-muted/50 transition-colors"
              data-testid="open-sidebar-btn"
            >
              <Menu className="w-5 h-5" />
            </button>
            
            {/* Persona Selector */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button 
                  variant="outline" 
                  className="rounded-none border-primary/30 bg-transparent hover:bg-primary/10 gap-2"
                  data-testid="persona-selector"
                >
                  <PersonaIcon className="w-4 h-4 text-primary" />
                  <span className="text-sm font-bold uppercase tracking-wider hidden sm:inline">
                    {selectedPersona?.name || "SELECT PERSONA"}
                  </span>
                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 bg-card border-border">
                <DropdownMenuLabel className="text-xs text-muted-foreground uppercase tracking-wider">
                  Select Persona
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                {personas.map((persona) => {
                  const Icon = PERSONA_ICONS[persona.icon] || Bot;
                  return (
                    <DropdownMenuItem
                      key={persona.id}
                      onClick={() => setSelectedPersona(persona)}
                      className={`cursor-pointer ${
                        selectedPersona?.id === persona.id ? 'bg-primary/20' : ''
                      }`}
                      data-testid={`persona-option-${persona.id}`}
                    >
                      <Icon className="w-4 h-4 mr-2 text-primary" />
                      <div className="flex-1">
                        <p className="font-bold text-sm">{persona.name}</p>
                        <p className="text-xs text-muted-foreground line-clamp-1">{persona.description}</p>
                      </div>
                    </DropdownMenuItem>
                  );
                })}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Tier Selector */}
            <Select value={selectedTier} onValueChange={setSelectedTier}>
              <SelectTrigger 
                className={`w-32 rounded-none border-primary/30 bg-transparent ${TIER_COLORS[selectedTier]}`}
                data-testid="tier-selector"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-card border-border">
                <SelectItem value="free" className="cursor-pointer">
                  <div className="flex items-center gap-2">
                    <Lock className="w-3 h-3" />
                    <span>Free</span>
                  </div>
                </SelectItem>
                <SelectItem value="pro" className="cursor-pointer">
                  <div className="flex items-center gap-2">
                    <Unlock className="w-3 h-3" />
                    <span>Pro</span>
                  </div>
                </SelectItem>
                <SelectItem value="dev" className="cursor-pointer">
                  <div className="flex items-center gap-2">
                    <Code className="w-3 h-3" />
                    <span>Dev+</span>
                  </div>
                </SelectItem>
                <SelectItem value="god" className="cursor-pointer">
                  <div className="flex items-center gap-2">
                    <Zap className="w-3 h-3" />
                    <span>God Mode</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Status Indicators */}
          <div className="hidden md:flex items-center gap-6">
            <Badge variant="outline" className={`text-xs ${
              systemStatus?.fusion_mode === 'Demo Mode' ? 'border-muted text-muted-foreground' : 'border-accent text-accent'
            }`}>
              {systemStatus?.fusion_mode || 'Loading...'}
            </Badge>
            <StatusIndicator enabled={systemStatus?.db_connected} label="DB" />
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Cpu className="w-3 h-3" />
              <span>{systemStatus?.total_messages || 0} msgs</span>
            </div>
          </div>
        </header>

        {/* Messages Area */}
        <ScrollArea className="flex-1 p-4 md:p-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
              <div className="w-24 h-24 mb-8 border border-primary/50 bg-primary/10 flex items-center justify-center neon-glow">
                <Layers className="w-12 h-12 text-primary" />
              </div>
              <h2 className="font-heading text-2xl md:text-3xl font-bold mb-4 text-primary neon-text">
                ECHELON CORE
              </h2>
              <p className="text-muted-foreground max-w-md mb-4 text-sm md:text-base">
                Trinity Fusion AI Framework - Multi-model synthesis for superior intelligence.
              </p>
              <p className="text-xs text-muted-foreground mb-8">
                {trinityStatus?.fusion_ready 
                  ? 'All systems operational. Trinity Fusion ready.'
                  : 'Running in demo mode. Configure API keys for full Trinity capabilities.'}
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {personas.slice(0, 4).map((persona) => {
                  const Icon = PERSONA_ICONS[persona.icon] || Bot;
                  return (
                    <button
                      key={persona.id}
                      onClick={() => setSelectedPersona(persona)}
                      className={`p-4 border transition-all duration-300 ${
                        selectedPersona?.id === persona.id
                          ? 'border-primary bg-primary/20 neon-glow'
                          : 'border-border hover:border-primary/50 hover:bg-muted/30'
                      }`}
                      data-testid={`quick-persona-${persona.id}`}
                    >
                      <Icon className="w-6 h-6 mx-auto mb-2 text-primary" />
                      <p className="text-xs font-bold uppercase tracking-wider">{persona.name}</p>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto">
              {messages.map((message) => (
                <MessageBubble 
                  key={message.id} 
                  message={message} 
                  persona={personas.find(p => p.id === message.persona_id) || selectedPersona}
                  fusionMode={message.fusion_mode}
                  modelsUsed={message.models_used}
                />
              ))}
              {isLoading && (
                <div className="flex gap-4 mb-6">
                  <div className="w-10 h-10 rounded-none flex items-center justify-center border bg-primary/20 border-primary/50 neon-glow">
                    <PersonaIcon className="w-5 h-5 text-primary animate-pulse" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs uppercase tracking-wider text-primary">
                        {selectedPersona?.name || 'GODMIND'}
                      </span>
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-primary/30 text-primary/70 animate-pulse">
                        Processing...
                      </Badge>
                    </div>
                    <div className="p-4 glass border-primary/30">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>

        {/* Input Area */}
        <div className="border-t border-border bg-card/50 backdrop-blur-sm p-4 md:p-6">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
                  placeholder="Enter command..."
                  disabled={isLoading}
                  className="w-full bg-black/50 border-b-2 border-muted hover:border-muted-foreground focus:border-primary px-4 py-3 text-foreground placeholder:text-muted-foreground font-mono transition-colors outline-none"
                  data-testid="chat-input"
                />
                <span className="absolute right-4 top-1/2 -translate-y-1/2 text-accent animate-blink">|</span>
              </div>
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="rounded-none border border-primary bg-primary/10 text-primary hover:bg-primary hover:text-black transition-all duration-300 px-6 disabled:opacity-50"
                data-testid="send-btn"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
            <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
              <span className="flex items-center gap-2">
                <Zap className="w-3 h-3 text-accent" />
                Trinity Fusion: Command R+ | DeepSeek | MythoMax
              </span>
              <span>Press Enter to send</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
