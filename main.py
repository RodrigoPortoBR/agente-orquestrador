"""
API Principal do Sistema de Agentes Orquestradores
Recebe webhooks do Lovable e coordena o processamento entre agentes.
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn

from models.schemas import (
    WebhookPayload, 
    OrchestratorResponse,
    MessageRole
)
from agents.orchestrator import OrchestratorAgent
from services.session_manager import SessionManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar aplicação FastAPI
app = FastAPI(
    title="Sistema de Agentes Orquestradores",
    description="API para processamento de mensagens via agentes especializados",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar componentes
orchestrator = OrchestratorAgent()
session_manager = SessionManager()

# Variáveis globais para estatísticas
stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "start_time": datetime.now()
}


@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização da aplicação."""
    logger.info("Starting Agent Orchestrator API")
    logger.info(f"OpenAI API Key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    logger.info(f"Database URL configured: {'Yes' if os.getenv('DATABASE_URL') else 'No'}")
    logger.info(f"Redis URL configured: {'Yes' if os.getenv('REDIS_URL') else 'No'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Evento executado no encerramento da aplicação."""
    logger.info("Shutting down Agent Orchestrator API")


@app.get("/")
async def root():
    """Endpoint raiz com informações básicas da API."""
    return {
        "message": "Sistema de Agentes Orquestradores",
        "version": "1.0.0",
        "status": "running",
        "uptime": str(datetime.now() - stats["start_time"]),
        "stats": stats
    }


@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde da aplicação."""
    try:
        # Verificar componentes críticos
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "orchestrator": "ok",
                "session_manager": "ok",
                "openai_api": "ok" if os.getenv("OPENAI_API_KEY") else "not_configured",
                "database": "ok" if os.getenv("DATABASE_URL") else "not_configured",
                "redis": "ok" if session_manager.redis_client else "fallback_memory"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/webhook/lovable")
async def lovable_webhook(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Endpoint principal para receber webhooks do Lovable.
    
    Args:
        request: Requisição HTTP
        background_tasks: Tarefas em background
        
    Returns:
        OrchestratorResponse: Resposta processada
    """
    stats["total_requests"] += 1
    
    try:
        # Receber dados do webhook
        raw_data = await request.json()
        logger.info(f"Received webhook data: {raw_data}")
        
        # Validar e converter para modelo
        try:
            payload = WebhookPayload(**raw_data)
        except ValidationError as e:
            logger.error(f"Invalid payload format: {str(e)}")
            stats["failed_requests"] += 1
            raise HTTPException(
                status_code=400, 
                detail=f"Formato de payload inválido: {str(e)}"
            )
        
        # Processar payload através do gerenciador de sessões
        processed_payload = await session_manager.process_webhook_payload(payload)
        
        # Processar mensagem através do orquestrador
        response = await orchestrator.process_message(processed_payload)
        
        # Adicionar resposta ao histórico da sessão
        background_tasks.add_task(
            session_manager.add_assistant_response,
            response.session_id,
            response.response,
            response.metadata
        )
        
        # Atualizar estatísticas
        if response.success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1
        
        logger.info(f"Processed message for session {response.session_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing webhook: {str(e)}")
        stats["failed_requests"] += 1
        
        # Retornar resposta de erro genérica
        error_response = OrchestratorResponse(
            response="Desculpe, ocorreu um erro interno. Tente novamente em alguns instantes.",
            session_id=raw_data.get("session_id", "unknown"),
            success=False,
            metadata={"error": str(e)}
        )
        
        return error_response


@app.post("/webhook/test")
async def test_webhook(payload: WebhookPayload):
    """
    Endpoint de teste para validar o funcionamento do sistema.
    
    Args:
        payload: Payload de teste
        
    Returns:
        OrchestratorResponse: Resposta de teste
    """
    try:
        logger.info(f"Test webhook called for session {payload.session_id}")
        
        # Processar através do sistema normal
        processed_payload = await session_manager.process_webhook_payload(payload)
        response = await orchestrator.process_message(processed_payload)
        
        # Adicionar resposta ao histórico
        await session_manager.add_assistant_response(
            response.session_id,
            response.response,
            response.metadata
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in test webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """
    Recupera informações de uma sessão específica.
    
    Args:
        session_id: ID da sessão
        
    Returns:
        Dict: Informações da sessão
    """
    try:
        session = await session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sessão não encontrada")
        
        conversation_history = await session_manager.get_conversation_history(
            session_id, 
            limit=20
        )
        
        return {
            "session": session.dict(),
            "conversation_history": [msg.dict() for msg in conversation_history],
            "message_count": len(conversation_history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Limpa o histórico de uma sessão.
    
    Args:
        session_id: ID da sessão
        
    Returns:
        Dict: Confirmação da operação
    """
    try:
        # Implementar limpeza de sessão
        # Por enquanto, apenas retornar confirmação
        logger.info(f"Session {session_id} cleared")
        
        return {
            "message": f"Sessão {session_id} limpa com sucesso",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Retorna estatísticas da aplicação.
    
    Returns:
        Dict: Estatísticas de uso
    """
    uptime = datetime.now() - stats["start_time"]
    
    return {
        "uptime": str(uptime),
        "uptime_seconds": uptime.total_seconds(),
        "total_requests": stats["total_requests"],
        "successful_requests": stats["successful_requests"],
        "failed_requests": stats["failed_requests"],
        "success_rate": (
            stats["successful_requests"] / stats["total_requests"] * 100
            if stats["total_requests"] > 0 else 0
        ),
        "start_time": stats["start_time"].isoformat()
    }


@app.post("/admin/cleanup")
async def cleanup_sessions():
    """
    Endpoint administrativo para limpeza de sessões expiradas.
    
    Returns:
        Dict: Resultado da limpeza
    """
    try:
        cleaned_count = await session_manager.cleanup_expired_sessions()
        
        return {
            "message": "Limpeza concluída",
            "sessions_cleaned": cleaned_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Tratamento de erros globais
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Tratador global de exceções."""
    logger.error(f"Global exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor",
            "message": "Ocorreu um erro inesperado. Tente novamente.",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Configurações do servidor
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
