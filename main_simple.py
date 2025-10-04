from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from typing import List, Optional
from datetime import datetime

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI(
    title="Agente Orquestrador Simples",
    description="Sistema básico de agentes para chat do Lovable",
    version="1.0.0"
)

# Modelos de dados
class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class WebhookPayload(BaseModel):
    session_id: str
    user_message: str
    conversation_history: Optional[List[ConversationMessage]] = []
    user_id: Optional[str] = None

class OrchestratorResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    success: bool

# Armazenamento simples em memória (para teste)
conversations = {}

@app.get("/")
def read_root():
    return {
        "message": "🤖 Agente Orquestrador funcionando!",
        "status": "online",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy",
        "openai_configured": openai_configured,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/webhook/lovable", response_model=OrchestratorResponse)
async def lovable_webhook(payload: WebhookPayload):
    try:
        # Verificar se OpenAI está configurada
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key não configurada")
        
        # Configurar cliente OpenAI
        client = openai.OpenAI(api_key=api_key)
        
        # Preparar histórico da conversa
        messages = [
            {
                "role": "system", 
                "content": """Você é um assistente inteligente que ajuda usuários com consultas e análises de dados.
                
                Características:
                - Responda sempre em português brasileiro
                - Seja amigável e prestativo
                - Se o usuário perguntar sobre dados ou consultas SQL, explique que você pode ajudar a analisar dados
                - Mantenha o contexto da conversa
                - Se não souber algo, seja honesto
                """
            }
        ]
        
        # Adicionar histórico da conversa
        for msg in payload.conversation_history[-5:]:  # Últimas 5 mensagens
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Adicionar mensagem atual
        messages.append({
            "role": "user",
            "content": payload.user_message
        })
        
        # Chamar OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Modelo mais barato
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        # Extrair resposta
        ai_response = response.choices[0].message.content
        
        # Salvar no histórico (simples)
        if payload.session_id not in conversations:
            conversations[payload.session_id] = []
        
        conversations[payload.session_id].extend([
            {"role": "user", "content": payload.user_message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": ai_response, "timestamp": datetime.now().isoformat()}
        ])
        
        return OrchestratorResponse(
            response=ai_response,
            session_id=payload.session_id,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
    except Exception as e:
        return OrchestratorResponse(
            response=f"Desculpe, ocorreu um erro: {str(e)}",
            session_id=payload.session_id,
            timestamp=datetime.now().isoformat(),
            success=False
        )

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Recuperar histórico de uma sessão"""
    history = conversations.get(session_id, [])
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history[-10:]  # Últimas 10 mensagens
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
