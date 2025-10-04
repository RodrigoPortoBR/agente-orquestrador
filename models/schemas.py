"""
Modelos de dados para o sistema de agentes orquestradores.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """Papéis possíveis para mensagens na conversa."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """Modelo para uma mensagem individual na conversa."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None


class WebhookPayload(BaseModel):
    """Payload recebido do webhook do Lovable."""
    session_id: str = Field(..., description="ID único da sessão do usuário")
    user_message: str = Field(..., description="Mensagem atual do usuário")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        description="Histórico da conversa"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SQLQueryRequest(BaseModel):
    """Requisição para o agente SQL."""
    query_intent: str = Field(..., description="Intenção da consulta em linguagem natural")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None
    session_id: str


class SQLQueryResponse(BaseModel):
    """Resposta do agente SQL."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    query_executed: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class OrchestratorResponse(BaseModel):
    """Resposta final do orquestrador para o Lovable."""
    response: str = Field(..., description="Resposta em linguagem natural")
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    metadata: Optional[Dict[str, Any]] = None


class AgentIntent(str, Enum):
    """Tipos de intenção identificados pelo orquestrador."""
    SQL_QUERY = "sql_query"
    GENERAL_CHAT = "general_chat"
    HELP = "help"
    UNKNOWN = "unknown"


class IntentAnalysis(BaseModel):
    """Resultado da análise de intenção."""
    intent: AgentIntent
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None


class SessionState(BaseModel):
    """Estado da sessão do usuário."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    conversation_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
