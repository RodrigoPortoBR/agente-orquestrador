"""
Agente Especializado em SQL - Responsável por converter solicitações em linguagem natural
para consultas SQL seguras e executá-las no banco de dados.
"""

import os
import re
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError

from models.schemas import SQLQueryRequest, SQLQueryResponse

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLQueryOutput(BaseModel):
    """Modelo para saída da geração de consulta SQL."""
    sql_query: str = Field(description="Consulta SQL gerada")
    explanation: str = Field(description="Explicação da consulta")
    estimated_safety: float = Field(description="Estimativa de segurança da consulta (0-1)")
    tables_used: List[str] = Field(description="Tabelas utilizadas na consulta")


class SQLAgent:
    """
    Agente especializado em análise e execução de consultas SQL.
    """
    
    def __init__(self):
        """Inicializa o agente SQL."""
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.0,  # Temperatura baixa para consultas SQL precisas
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Parser para geração de SQL
        self.sql_parser = JsonOutputParser(pydantic_object=SQLQueryOutput)
        
        # Template para geração de SQL
        self.sql_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_sql_system_prompt()),
            HumanMessage(content="""
            Schema do banco de dados:
            {database_schema}
            
            Solicitação do usuário: {user_request}
            Parâmetros extraídos: {parameters}
            
            Gere uma consulta SQL segura e eficiente para atender à solicitação.
            """)
        ])
        
        # Configurar conexão com banco de dados
        self.database_url = os.getenv("DATABASE_URL")
        self.engine = None
        self.metadata = None
        
        if self.database_url:
            try:
                self.engine = create_engine(self.database_url)
                self.metadata = MetaData()
                self.metadata.reflect(bind=self.engine)
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Failed to connect to database: {str(e)}")
        
        # Lista de palavras-chave perigosas
        self.dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
            'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE'
        ]
    
    def _get_sql_system_prompt(self) -> str:
        """Retorna o prompt do sistema para geração de SQL."""
        return """
        Você é um especialista em SQL responsável por converter solicitações em linguagem natural
        para consultas SQL seguras e eficientes.
        
        REGRAS CRÍTICAS DE SEGURANÇA:
        1. APENAS consultas SELECT são permitidas
        2. NUNCA use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE
        3. Use LIMIT para evitar consultas que retornem muitos dados
        4. Valide todos os parâmetros de entrada
        5. Use aspas adequadas para nomes de colunas e tabelas
        6. Evite consultas que possam causar sobrecarga no banco
        
        DIRETRIZES TÉCNICAS:
        1. Analise o schema fornecido cuidadosamente
        2. Use JOINs apropriados quando necessário
        3. Aplique filtros baseados nos parâmetros extraídos
        4. Use funções de agregação quando apropriado (COUNT, SUM, AVG, etc.)
        5. Ordene resultados de forma lógica
        6. Limite resultados a no máximo 100 registros por padrão
        
        FORMATO DE RESPOSTA:
        - Retorne SEMPRE um JSON válido
        - Inclua a consulta SQL, explicação, estimativa de segurança e tabelas usadas
        - A estimativa de segurança deve ser 1.0 apenas para consultas SELECT simples
        - Se a solicitação for perigosa ou impossível, retorne estimativa 0.0
        
        Responda sempre em português brasileiro.
        """
    
    def _get_database_schema(self) -> str:
        """
        Obtém o schema do banco de dados.
        
        Returns:
            str: Descrição do schema em formato texto
        """
        if not self.engine or not self.metadata:
            return """
            # Schema de Exemplo (para demonstração)
            
            ## Tabela: usuarios
            - id (INTEGER, PRIMARY KEY)
            - nome (VARCHAR)
            - email (VARCHAR)
            - data_cadastro (TIMESTAMP)
            - ativo (BOOLEAN)
            
            ## Tabela: vendas
            - id (INTEGER, PRIMARY KEY)
            - usuario_id (INTEGER, FOREIGN KEY)
            - produto (VARCHAR)
            - valor (DECIMAL)
            - data_venda (TIMESTAMP)
            - status (VARCHAR)
            
            ## Tabela: produtos
            - id (INTEGER, PRIMARY KEY)
            - nome (VARCHAR)
            - categoria (VARCHAR)
            - preco (DECIMAL)
            - estoque (INTEGER)
            """
        
        try:
            inspector = inspect(self.engine)
            schema_description = "# Schema do Banco de Dados\n\n"
            
            for table_name in inspector.get_table_names():
                schema_description += f"## Tabela: {table_name}\n"
                columns = inspector.get_columns(table_name)
                
                for column in columns:
                    column_info = f"- {column['name']} ({column['type']}"
                    if column.get('nullable', True) is False:
                        column_info += ", NOT NULL"
                    if column.get('primary_key', False):
                        column_info += ", PRIMARY KEY"
                    column_info += ")\n"
                    schema_description += column_info
                
                # Adicionar informações sobre chaves estrangeiras
                foreign_keys = inspector.get_foreign_keys(table_name)
                for fk in foreign_keys:
                    schema_description += f"- {fk['constrained_columns'][0]} (FOREIGN KEY -> {fk['referred_table']}.{fk['referred_columns'][0]})\n"
                
                schema_description += "\n"
            
            return schema_description
            
        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
            return "Erro ao obter schema do banco de dados"
    
    def _validate_sql_safety(self, sql_query: str) -> Tuple[bool, str]:
        """
        Valida se a consulta SQL é segura.
        
        Args:
            sql_query: Consulta SQL para validar
            
        Returns:
            Tuple[bool, str]: (é_segura, motivo)
        """
        # Converter para maiúsculo para análise
        sql_upper = sql_query.upper().strip()
        
        # Verificar se começa com SELECT
        if not sql_upper.startswith('SELECT'):
            return False, "Apenas consultas SELECT são permitidas"
        
        # Verificar palavras-chave perigosas
        for keyword in self.dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Palavra-chave perigosa detectada: {keyword}"
        
        # Verificar se há múltiplas declarações (SQL injection básico)
        if ';' in sql_query and not sql_query.strip().endswith(';'):
            return False, "Múltiplas declarações SQL não são permitidas"
        
        # Verificar comentários suspeitos
        if '--' in sql_query or '/*' in sql_query:
            return False, "Comentários SQL não são permitidos"
        
        return True, "Consulta aprovada na validação de segurança"
    
    async def generate_sql_query(
        self, 
        user_request: str, 
        parameters: Dict[str, Any]
    ) -> SQLQueryOutput:
        """
        Gera uma consulta SQL baseada na solicitação do usuário.
        
        Args:
            user_request: Solicitação em linguagem natural
            parameters: Parâmetros extraídos pelo orquestrador
            
        Returns:
            SQLQueryOutput: Consulta SQL gerada
        """
        try:
            # Obter schema do banco
            database_schema = self._get_database_schema()
            
            # Executar geração de SQL
            chain = self.sql_prompt | self.llm | self.sql_parser
            
            result = await chain.ainvoke({
                "database_schema": database_schema,
                "user_request": user_request,
                "parameters": parameters
            })
            
            # Criar objeto de saída
            sql_output = SQLQueryOutput(
                sql_query=result["sql_query"],
                explanation=result["explanation"],
                estimated_safety=result["estimated_safety"],
                tables_used=result["tables_used"]
            )
            
            logger.info(f"SQL query generated with safety score: {sql_output.estimated_safety}")
            return sql_output
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            return SQLQueryOutput(
                sql_query="",
                explanation=f"Erro na geração da consulta: {str(e)}",
                estimated_safety=0.0,
                tables_used=[]
            )
    
    def execute_sql_query(self, sql_query: str) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str], float]:
        """
        Executa uma consulta SQL no banco de dados.
        
        Args:
            sql_query: Consulta SQL para executar
            
        Returns:
            Tuple[bool, Optional[List[Dict]], Optional[str], float]: 
            (sucesso, dados, erro, tempo_execução)
        """
        if not self.engine:
            return False, None, "Conexão com banco de dados não configurada", 0.0
        
        # Validar segurança da consulta
        is_safe, safety_reason = self._validate_sql_safety(sql_query)
        if not is_safe:
            return False, None, f"Consulta rejeitada: {safety_reason}", 0.0
        
        try:
            start_time = time.time()
            
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query))
                
                # Converter resultado para lista de dicionários
                columns = result.keys()
                data = []
                
                for row in result.fetchall():
                    row_dict = {}
                    for i, column in enumerate(columns):
                        value = row[i]
                        # Converter tipos especiais para JSON serializável
                        if isinstance(value, datetime):
                            value = value.isoformat()
                        elif hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool)):
                            value = str(value)
                        row_dict[column] = value
                    data.append(row_dict)
                
                execution_time = time.time() - start_time
                logger.info(f"SQL query executed successfully in {execution_time:.3f}s, returned {len(data)} rows")
                
                return True, data, None, execution_time
                
        except SQLAlchemyError as e:
            logger.error(f"SQL execution error: {str(e)}")
            return False, None, f"Erro na execução da consulta: {str(e)}", 0.0
        except Exception as e:
            logger.error(f"Unexpected error executing SQL: {str(e)}")
            return False, None, f"Erro inesperado: {str(e)}", 0.0
    
    async def execute_query(
        self, 
        query_intent: str, 
        parameters: Dict[str, Any], 
        session_id: str,
        context: Optional[str] = None
    ) -> SQLQueryResponse:
        """
        Executa todo o fluxo de processamento de uma consulta.
        
        Args:
            query_intent: Intenção da consulta em linguagem natural
            parameters: Parâmetros extraídos
            session_id: ID da sessão
            context: Contexto adicional
            
        Returns:
            SQLQueryResponse: Resposta completa da consulta
        """
        try:
            logger.info(f"Processing SQL query for session {session_id}")
            
            # 1. Gerar consulta SQL
            sql_output = await self.generate_sql_query(query_intent, parameters)
            
            # 2. Verificar se a consulta é segura o suficiente
            if sql_output.estimated_safety < 0.7:
                return SQLQueryResponse(
                    success=False,
                    error=f"Consulta considerada insegura: {sql_output.explanation}",
                    query_executed=sql_output.sql_query,
                    metadata={
                        "safety_score": sql_output.estimated_safety,
                        "tables_used": sql_output.tables_used
                    }
                )
            
            # 3. Executar consulta
            success, data, error, execution_time = self.execute_sql_query(sql_output.sql_query)
            
            return SQLQueryResponse(
                success=success,
                data=data,
                error=error,
                query_executed=sql_output.sql_query,
                execution_time=execution_time,
                metadata={
                    "explanation": sql_output.explanation,
                    "safety_score": sql_output.estimated_safety,
                    "tables_used": sql_output.tables_used,
                    "session_id": session_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error in execute_query: {str(e)}")
            return SQLQueryResponse(
                success=False,
                error=f"Erro no processamento da consulta: {str(e)}",
                metadata={"session_id": session_id}
            )
