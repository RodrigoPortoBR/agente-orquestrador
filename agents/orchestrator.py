"""
Agente Orquestrador Principal - Responsável por analisar mensagens do usuário,
identificar intenções e coordenar com outros agentes especializados.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from models.schemas import (
    WebhookPayload, 
    OrchestratorResponse, 
    AgentIntent, 
    IntentAnalysis,
    ConversationMessage,
    MessageRole
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentAnalysisOutput(BaseModel):
    """Modelo para saída da análise de intenção."""
    intent: str = Field(description="Tipo de intenção identificada")
    confidence: float = Field(description="Confiança na identificação (0-1)")
    extracted_parameters: Dict[str, Any] = Field(description="Parâmetros extraídos da mensagem")
    reasoning: str = Field(description="Explicação do raciocínio")


class OrchestratorAgent:
    """
    Agente Orquestrador Principal que coordena todo o fluxo de processamento.
    """
    
    def __init__(self):
        """Inicializa o agente orquestrador."""
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Parser para análise de intenção
        self.intent_parser = JsonOutputParser(pydantic_object=IntentAnalysisOutput)
        
        # Template para análise de intenção
        self.intent_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_intent_system_prompt()),
            MessagesPlaceholder(variable_name="conversation_history"),
            HumanMessage(content="Mensagem atual: {user_message}")
        ])
        
        # Template para geração de resposta final
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_response_system_prompt()),
            MessagesPlaceholder(variable_name="conversation_history"),
            HumanMessage(content="""
            Dados da consulta SQL: {sql_data}
            Mensagem original do usuário: {user_message}
            
            Gere uma resposta natural e informativa baseada nos dados retornados.
            """)
        ])
    
    def _get_intent_system_prompt(self) -> str:
        """Retorna o prompt do sistema para análise de intenção."""
        return """
        Você é um assistente especializado em analisar intenções de usuários em um sistema de chat.
        Sua função é identificar o que o usuário deseja fazer e extrair parâmetros relevantes.
        
        Tipos de intenção possíveis:
        - "sql_query": Usuário quer fazer uma consulta ao banco de dados (ex: "quantos usuários temos?", "mostre as vendas do mês")
        - "general_chat": Conversa geral, saudações, perguntas sobre o sistema
        - "help": Usuário precisa de ajuda ou instruções
        - "unknown": Não foi possível identificar a intenção
        
        Para intenções "sql_query", extraia parâmetros como:
        - tabelas mencionadas
        - campos de interesse
        - filtros (datas, categorias, etc.)
        - tipo de agregação (soma, contagem, média, etc.)
        
        Responda SEMPRE em formato JSON válido seguindo o schema especificado.
        Seja preciso na identificação da intenção e confiante na sua análise.
        """
    
    def _get_response_system_prompt(self) -> str:
        """Retorna o prompt do sistema para geração de resposta."""
        return """
        Você é um assistente que converte dados estruturados em respostas naturais e informativas.
        
        Diretrizes:
        1. Seja claro e objetivo na resposta
        2. Use linguagem natural e amigável
        3. Se houver dados numéricos, apresente-os de forma organizada
        4. Se não houver dados, explique o motivo de forma educada
        5. Mantenha o contexto da conversa
        6. Use formatação quando apropriado (listas, tabelas simples)
        
        Sempre responda em português brasileiro.
        """
    
    async def analyze_intent(self, payload: WebhookPayload) -> IntentAnalysis:
        """
        Analisa a intenção da mensagem do usuário.
        
        Args:
            payload: Dados recebidos do webhook
            
        Returns:
            IntentAnalysis: Resultado da análise de intenção
        """
        try:
            # Converter histórico para formato do LangChain
            messages = self._convert_history_to_messages(payload.conversation_history)
            
            # Executar análise de intenção
            chain = self.intent_prompt | self.llm | self.intent_parser
            
            result = await chain.ainvoke({
                "conversation_history": messages,
                "user_message": payload.user_message
            })
            
            # Converter resultado para nosso modelo
            intent_analysis = IntentAnalysis(
                intent=AgentIntent(result["intent"]),
                confidence=result["confidence"],
                extracted_parameters=result["extracted_parameters"],
                reasoning=result["reasoning"]
            )
            
            logger.info(f"Intent analyzed: {intent_analysis.intent} (confidence: {intent_analysis.confidence})")
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            return IntentAnalysis(
                intent=AgentIntent.UNKNOWN,
                confidence=0.0,
                extracted_parameters={},
                reasoning=f"Erro na análise: {str(e)}"
            )
    
    async def generate_response(
        self, 
        user_message: str, 
        sql_data: Optional[Dict[str, Any]], 
        conversation_history: List[ConversationMessage]
    ) -> str:
        """
        Gera uma resposta em linguagem natural baseada nos dados SQL.
        
        Args:
            user_message: Mensagem original do usuário
            sql_data: Dados retornados pela consulta SQL
            conversation_history: Histórico da conversa
            
        Returns:
            str: Resposta em linguagem natural
        """
        try:
            # Converter histórico para formato do LangChain
            messages = self._convert_history_to_messages(conversation_history)
            
            # Preparar dados SQL para o prompt
            sql_data_str = json.dumps(sql_data, indent=2, ensure_ascii=False) if sql_data else "Nenhum dado disponível"
            
            # Executar geração de resposta
            chain = self.response_prompt | self.llm
            
            result = await chain.ainvoke({
                "conversation_history": messages,
                "sql_data": sql_data_str,
                "user_message": user_message
            })
            
            return result.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Desculpe, ocorreu um erro ao processar sua solicitação: {str(e)}"
    
    def _convert_history_to_messages(self, history: List[ConversationMessage]) -> List:
        """
        Converte histórico de conversa para formato do LangChain.
        
        Args:
            history: Lista de mensagens da conversa
            
        Returns:
            List: Mensagens no formato do LangChain
        """
        messages = []
        
        for msg in history:
            if msg.role == MessageRole.USER:
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                messages.append(SystemMessage(content=msg.content))
        
        return messages
    
    async def process_message(self, payload: WebhookPayload) -> OrchestratorResponse:
        """
        Processa uma mensagem completa do usuário.
        
        Args:
            payload: Dados recebidos do webhook
            
        Returns:
            OrchestratorResponse: Resposta processada
        """
        try:
            logger.info(f"Processing message for session {payload.session_id}")
            
            # 1. Analisar intenção
            intent_analysis = await self.analyze_intent(payload)
            
            # 2. Processar baseado na intenção
            if intent_analysis.intent == AgentIntent.SQL_QUERY:
                # Importar aqui para evitar dependência circular
                from agents.sql_agent import SQLAgent
                
                sql_agent = SQLAgent()
                sql_response = await sql_agent.execute_query(
                    query_intent=payload.user_message,
                    parameters=intent_analysis.extracted_parameters,
                    session_id=payload.session_id
                )
                
                # Gerar resposta natural
                response_text = await self.generate_response(
                    user_message=payload.user_message,
                    sql_data=sql_response.data if sql_response.success else None,
                    conversation_history=payload.conversation_history
                )
                
            elif intent_analysis.intent == AgentIntent.GENERAL_CHAT:
                response_text = await self._handle_general_chat(payload)
                
            elif intent_analysis.intent == AgentIntent.HELP:
                response_text = self._get_help_response()
                
            else:
                response_text = "Desculpe, não consegui entender sua solicitação. Pode reformular a pergunta?"
            
            return OrchestratorResponse(
                response=response_text,
                session_id=payload.session_id,
                success=True,
                metadata={
                    "intent": intent_analysis.intent.value,
                    "confidence": intent_analysis.confidence
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return OrchestratorResponse(
                response=f"Ocorreu um erro interno. Tente novamente em alguns instantes.",
                session_id=payload.session_id,
                success=False,
                metadata={"error": str(e)}
            )
    
    async def _handle_general_chat(self, payload: WebhookPayload) -> str:
        """Lida com conversas gerais."""
        # Implementação simples para chat geral
        general_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            Você é um assistente amigável que ajuda usuários com um sistema de consulta de dados.
            Responda de forma natural e educada. Se o usuário fizer perguntas sobre dados,
            sugira que ele faça uma pergunta específica sobre os dados que deseja consultar.
            """),
            MessagesPlaceholder(variable_name="conversation_history"),
            HumanMessage(content="{user_message}")
        ])
        
        messages = self._convert_history_to_messages(payload.conversation_history)
        chain = general_prompt | self.llm
        
        result = await chain.ainvoke({
            "conversation_history": messages,
            "user_message": payload.user_message
        })
        
        return result.content
    
    def _get_help_response(self) -> str:
        """Retorna mensagem de ajuda."""
        return """
        Olá! Eu sou seu assistente para consultas de dados. Posso ajudá-lo com:
        
        📊 **Consultas de dados**: Faça perguntas sobre seus dados, como:
        - "Quantos usuários temos cadastrados?"
        - "Mostre as vendas do último mês"
        - "Qual produto mais vendido?"
        
        💬 **Conversa geral**: Posso responder perguntas sobre como usar o sistema
        
        🔍 **Dicas**: Seja específico em suas perguntas para obter melhores resultados!
        
        Como posso ajudá-lo hoje?
        """
