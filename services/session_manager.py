"""
Gerenciador de Sessões - Responsável por manter o estado das conversas
e gerenciar a memória conversacional dos usuários.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError

from models.schemas import (
    SessionState, 
    ConversationMessage, 
    MessageRole,
    WebhookPayload
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Gerenciador de sessões e memória conversacional.
    """
    
    def __init__(self):
        """Inicializa o gerenciador de sessões."""
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        
        # Configurações
        self.session_ttl = 3600 * 24  # 24 horas
        self.max_conversation_history = 50  # Máximo de mensagens por sessão
        self.conversation_context_limit = 10  # Mensagens enviadas para o LLM
        
        # Tentar conectar ao Redis
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory storage: {str(e)}")
            self.redis_client = None
            self._memory_storage = {}  # Fallback para armazenamento em memória
    
    def _get_session_key(self, session_id: str) -> str:
        """Gera chave para sessão no Redis."""
        return f"session:{session_id}"
    
    def _get_conversation_key(self, session_id: str) -> str:
        """Gera chave para histórico de conversa no Redis."""
        return f"conversation:{session_id}"
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Recupera o estado de uma sessão.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            Optional[SessionState]: Estado da sessão ou None se não existir
        """
        try:
            if self.redis_client:
                session_data = self.redis_client.get(self._get_session_key(session_id))
                if session_data:
                    data = json.loads(session_data)
                    return SessionState(**data)
            else:
                # Fallback para memória
                session_data = self._memory_storage.get(self._get_session_key(session_id))
                if session_data:
                    return SessionState(**session_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {str(e)}")
            return None
    
    async def create_session(
        self, 
        session_id: str, 
        user_id: Optional[str] = None
    ) -> SessionState:
        """
        Cria uma nova sessão.
        
        Args:
            session_id: ID da sessão
            user_id: ID do usuário (opcional)
            
        Returns:
            SessionState: Nova sessão criada
        """
        try:
            session = SessionState(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                conversation_count=0,
                context={},
                is_active=True
            )
            
            await self.save_session(session)
            logger.info(f"Created new session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {str(e)}")
            raise
    
    async def save_session(self, session: SessionState) -> bool:
        """
        Salva o estado de uma sessão.
        
        Args:
            session: Estado da sessão para salvar
            
        Returns:
            bool: True se salvou com sucesso
        """
        try:
            session_data = session.dict()
            
            # Converter datetime para string
            for key, value in session_data.items():
                if isinstance(value, datetime):
                    session_data[key] = value.isoformat()
            
            if self.redis_client:
                self.redis_client.setex(
                    self._get_session_key(session.session_id),
                    self.session_ttl,
                    json.dumps(session_data)
                )
            else:
                # Fallback para memória
                self._memory_storage[self._get_session_key(session.session_id)] = session_data
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")
            return False
    
    async def get_conversation_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """
        Recupera o histórico de conversa de uma sessão.
        
        Args:
            session_id: ID da sessão
            limit: Limite de mensagens (padrão: conversation_context_limit)
            
        Returns:
            List[ConversationMessage]: Histórico da conversa
        """
        try:
            if limit is None:
                limit = self.conversation_context_limit
            
            if self.redis_client:
                # Recuperar mensagens mais recentes
                messages_data = self.redis_client.lrange(
                    self._get_conversation_key(session_id), 
                    -limit, 
                    -1
                )
                
                messages = []
                for msg_data in messages_data:
                    msg_dict = json.loads(msg_data)
                    # Converter timestamp string de volta para datetime
                    if 'timestamp' in msg_dict and isinstance(msg_dict['timestamp'], str):
                        msg_dict['timestamp'] = datetime.fromisoformat(msg_dict['timestamp'])
                    messages.append(ConversationMessage(**msg_dict))
                
                return messages
            else:
                # Fallback para memória
                conversation_key = self._get_conversation_key(session_id)
                messages_data = self._memory_storage.get(conversation_key, [])
                
                messages = []
                for msg_dict in messages_data[-limit:]:
                    if isinstance(msg_dict['timestamp'], str):
                        msg_dict['timestamp'] = datetime.fromisoformat(msg_dict['timestamp'])
                    messages.append(ConversationMessage(**msg_dict))
                
                return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history for {session_id}: {str(e)}")
            return []
    
    async def add_message(
        self, 
        session_id: str, 
        role: MessageRole, 
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Adiciona uma mensagem ao histórico da conversa.
        
        Args:
            session_id: ID da sessão
            role: Papel da mensagem (user, assistant, system)
            content: Conteúdo da mensagem
            metadata: Metadados adicionais
            
        Returns:
            bool: True se adicionou com sucesso
        """
        try:
            message = ConversationMessage(
                role=role,
                content=content,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            message_data = message.dict()
            # Converter datetime para string
            message_data['timestamp'] = message_data['timestamp'].isoformat()
            
            if self.redis_client:
                conversation_key = self._get_conversation_key(session_id)
                
                # Adicionar mensagem à lista
                self.redis_client.lpush(conversation_key, json.dumps(message_data))
                
                # Manter apenas as últimas N mensagens
                self.redis_client.ltrim(conversation_key, 0, self.max_conversation_history - 1)
                
                # Definir TTL para a conversa
                self.redis_client.expire(conversation_key, self.session_ttl)
            else:
                # Fallback para memória
                conversation_key = self._get_conversation_key(session_id)
                if conversation_key not in self._memory_storage:
                    self._memory_storage[conversation_key] = []
                
                self._memory_storage[conversation_key].insert(0, message_data)
                
                # Manter apenas as últimas N mensagens
                if len(self._memory_storage[conversation_key]) > self.max_conversation_history:
                    self._memory_storage[conversation_key] = self._memory_storage[conversation_key][:self.max_conversation_history]
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {str(e)}")
            return False
    
    async def update_session_activity(self, session_id: str) -> bool:
        """
        Atualiza a última atividade de uma sessão.
        
        Args:
            session_id: ID da sessão
            
        Returns:
            bool: True se atualizou com sucesso
        """
        try:
            session = await self.get_session(session_id)
            if session:
                session.last_activity = datetime.now()
                session.conversation_count += 1
                return await self.save_session(session)
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating session activity {session_id}: {str(e)}")
            return False
    
    async def process_webhook_payload(self, payload: WebhookPayload) -> WebhookPayload:
        """
        Processa um payload de webhook, gerenciando sessão e histórico.
        
        Args:
            payload: Payload recebido do webhook
            
        Returns:
            WebhookPayload: Payload processado com histórico atualizado
        """
        try:
            # 1. Verificar se sessão existe, criar se necessário
            session = await self.get_session(payload.session_id)
            if not session:
                session = await self.create_session(
                    payload.session_id, 
                    payload.user_id
                )
            
            # 2. Adicionar mensagem do usuário ao histórico
            await self.add_message(
                payload.session_id,
                MessageRole.USER,
                payload.user_message,
                payload.metadata
            )
            
            # 3. Recuperar histórico atualizado
            conversation_history = await self.get_conversation_history(payload.session_id)
            
            # 4. Atualizar atividade da sessão
            await self.update_session_activity(payload.session_id)
            
            # 5. Atualizar payload com histórico
            payload.conversation_history = conversation_history
            
            return payload
            
        except Exception as e:
            logger.error(f"Error processing webhook payload: {str(e)}")
            # Retornar payload original em caso de erro
            return payload
    
    async def add_assistant_response(
        self, 
        session_id: str, 
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Adiciona uma resposta do assistente ao histórico.
        
        Args:
            session_id: ID da sessão
            response: Resposta do assistente
            metadata: Metadados da resposta
            
        Returns:
            bool: True se adicionou com sucesso
        """
        return await self.add_message(
            session_id,
            MessageRole.ASSISTANT,
            response,
            metadata
        )
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Remove sessões expiradas (apenas para fallback em memória).
        
        Returns:
            int: Número de sessões removidas
        """
        if self.redis_client:
            # Redis gerencia TTL automaticamente
            return 0
        
        try:
            expired_count = 0
            current_time = datetime.now()
            expired_keys = []
            
            for key, session_data in self._memory_storage.items():
                if key.startswith('session:'):
                    if 'last_activity' in session_data:
                        last_activity = datetime.fromisoformat(session_data['last_activity'])
                        if current_time - last_activity > timedelta(seconds=self.session_ttl):
                            expired_keys.append(key)
                            # Também remover conversa associada
                            conversation_key = key.replace('session:', 'conversation:')
                            if conversation_key in self._memory_storage:
                                expired_keys.append(conversation_key)
            
            for key in expired_keys:
                del self._memory_storage[key]
                expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired sessions")
            
            return expired_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {str(e)}")
            return 0
