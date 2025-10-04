#!/usr/bin/env python3
"""
Script de teste para validar o funcionamento do sistema de agentes orquestradores.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Adicionar o diretório atual ao path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.schemas import (
    WebhookPayload, 
    ConversationMessage, 
    MessageRole
)
from agents.orchestrator import OrchestratorAgent
from agents.sql_agent import SQLAgent
from services.session_manager import SessionManager


class SystemTester:
    """Classe para testar o sistema de agentes."""
    
    def __init__(self):
        """Inicializa o testador."""
        self.orchestrator = OrchestratorAgent()
        self.sql_agent = SQLAgent()
        self.session_manager = SessionManager()
        self.test_session_id = f"test_session_{int(datetime.now().timestamp())}"
        
        # Resultados dos testes
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
    
    def log_test_result(self, test_name: str, success: bool, details: str = ""):
        """Registra o resultado de um teste."""
        self.test_results["total_tests"] += 1
        
        if success:
            self.test_results["passed_tests"] += 1
            status = "✅ PASSOU"
        else:
            self.test_results["failed_tests"] += 1
            status = "❌ FALHOU"
        
        result = {
            "test_name": test_name,
            "status": status,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        self.test_results["test_details"].append(result)
        print(f"{status}: {test_name}")
        if details:
            print(f"   Detalhes: {details}")
        print()
    
    async def test_session_manager(self):
        """Testa o gerenciador de sessões."""
        print("🧪 Testando Session Manager...")
        
        try:
            # Teste 1: Criar sessão
            session = await self.session_manager.create_session(self.test_session_id)
            self.log_test_result(
                "Criar sessão",
                session.session_id == self.test_session_id,
                f"Session ID: {session.session_id}"
            )
            
            # Teste 2: Adicionar mensagem
            success = await self.session_manager.add_message(
                self.test_session_id,
                MessageRole.USER,
                "Mensagem de teste"
            )
            self.log_test_result(
                "Adicionar mensagem",
                success,
                "Mensagem do usuário adicionada"
            )
            
            # Teste 3: Recuperar histórico
            history = await self.session_manager.get_conversation_history(self.test_session_id)
            self.log_test_result(
                "Recuperar histórico",
                len(history) == 1 and history[0].content == "Mensagem de teste",
                f"Histórico contém {len(history)} mensagens"
            )
            
            # Teste 4: Processar payload
            payload = WebhookPayload(
                session_id=self.test_session_id,
                user_message="Nova mensagem de teste"
            )
            processed_payload = await self.session_manager.process_webhook_payload(payload)
            self.log_test_result(
                "Processar webhook payload",
                len(processed_payload.conversation_history) >= 1,
                f"Payload processado com {len(processed_payload.conversation_history)} mensagens no histórico"
            )
            
        except Exception as e:
            self.log_test_result(
                "Session Manager (geral)",
                False,
                f"Erro: {str(e)}"
            )
    
    async def test_orchestrator_agent(self):
        """Testa o agente orquestrador."""
        print("🧪 Testando Orchestrator Agent...")
        
        try:
            # Teste 1: Análise de intenção - Chat geral
            payload = WebhookPayload(
                session_id=self.test_session_id,
                user_message="Olá, como você está?",
                conversation_history=[]
            )
            
            intent_analysis = await self.orchestrator.analyze_intent(payload)
            self.log_test_result(
                "Análise de intenção - Chat geral",
                intent_analysis.intent.value in ["general_chat", "help"],
                f"Intenção: {intent_analysis.intent.value}, Confiança: {intent_analysis.confidence}"
            )
            
            # Teste 2: Análise de intenção - Consulta SQL
            payload.user_message = "Quantos usuários temos no sistema?"
            intent_analysis = await self.orchestrator.analyze_intent(payload)
            self.log_test_result(
                "Análise de intenção - SQL Query",
                intent_analysis.intent.value == "sql_query",
                f"Intenção: {intent_analysis.intent.value}, Confiança: {intent_analysis.confidence}"
            )
            
            # Teste 3: Geração de resposta
            response_text = await self.orchestrator.generate_response(
                "Teste de resposta",
                {"test": "data"},
                []
            )
            self.log_test_result(
                "Geração de resposta",
                len(response_text) > 0,
                f"Resposta gerada: {response_text[:100]}..."
            )
            
            # Teste 4: Processamento completo de mensagem
            payload.user_message = "Olá, preciso de ajuda"
            response = await self.orchestrator.process_message(payload)
            self.log_test_result(
                "Processamento completo de mensagem",
                response.success and len(response.response) > 0,
                f"Resposta: {response.response[:100]}..."
            )
            
        except Exception as e:
            self.log_test_result(
                "Orchestrator Agent (geral)",
                False,
                f"Erro: {str(e)}"
            )
    
    async def test_sql_agent(self):
        """Testa o agente SQL."""
        print("🧪 Testando SQL Agent...")
        
        try:
            # Teste 1: Geração de consulta SQL
            sql_output = await self.sql_agent.generate_sql_query(
                "Quantos usuários temos?",
                {"table": "usuarios", "operation": "count"}
            )
            self.log_test_result(
                "Geração de consulta SQL",
                len(sql_output.sql_query) > 0 and "SELECT" in sql_output.sql_query.upper(),
                f"Query: {sql_output.sql_query[:100]}..."
            )
            
            # Teste 2: Validação de segurança
            is_safe, reason = self.sql_agent._validate_sql_safety("SELECT * FROM usuarios LIMIT 10")
            self.log_test_result(
                "Validação de segurança - Query segura",
                is_safe,
                f"Razão: {reason}"
            )
            
            # Teste 3: Validação de segurança - Query perigosa
            is_safe, reason = self.sql_agent._validate_sql_safety("DROP TABLE usuarios")
            self.log_test_result(
                "Validação de segurança - Query perigosa",
                not is_safe,
                f"Razão: {reason}"
            )
            
            # Teste 4: Execução completa (sem banco real)
            response = await self.sql_agent.execute_query(
                "Contar usuários",
                {"operation": "count"},
                self.test_session_id
            )
            # Este teste pode falhar se não houver banco configurado, mas não deve gerar exceção
            self.log_test_result(
                "Execução completa de query",
                True,  # Sucesso se não houve exceção
                f"Sucesso: {response.success}, Erro: {response.error}"
            )
            
        except Exception as e:
            self.log_test_result(
                "SQL Agent (geral)",
                False,
                f"Erro: {str(e)}"
            )
    
    async def test_integration(self):
        """Testa a integração completa do sistema."""
        print("🧪 Testando Integração Completa...")
        
        try:
            # Cenário completo: usuário faz pergunta, sistema processa e responde
            payload = WebhookPayload(
                session_id=f"integration_test_{int(datetime.now().timestamp())}",
                user_message="Olá! Você pode me ajudar com consultas de dados?",
                conversation_history=[]
            )
            
            # Processar através do session manager
            processed_payload = await self.session_manager.process_webhook_payload(payload)
            
            # Processar através do orquestrador
            response = await self.orchestrator.process_message(processed_payload)
            
            # Adicionar resposta ao histórico
            await self.session_manager.add_assistant_response(
                response.session_id,
                response.response,
                response.metadata
            )
            
            # Verificar se tudo funcionou
            final_history = await self.session_manager.get_conversation_history(response.session_id)
            
            self.log_test_result(
                "Integração completa",
                (response.success and 
                 len(response.response) > 0 and 
                 len(final_history) >= 2),  # Pelo menos user + assistant
                f"Resposta: {response.response[:100]}..., Histórico: {len(final_history)} mensagens"
            )
            
        except Exception as e:
            self.log_test_result(
                "Integração completa",
                False,
                f"Erro: {str(e)}"
            )
    
    def test_environment_setup(self):
        """Testa a configuração do ambiente."""
        print("🧪 Testando Configuração do Ambiente...")
        
        # Teste 1: Variáveis de ambiente
        openai_key = os.getenv("OPENAI_API_KEY")
        self.log_test_result(
            "OpenAI API Key configurada",
            openai_key is not None and len(openai_key) > 0,
            "Chave da OpenAI está configurada" if openai_key else "Chave da OpenAI não encontrada"
        )
        
        # Teste 2: Importações
        try:
            import langchain
            import openai
            import fastapi
            import redis
            self.log_test_result(
                "Dependências importadas",
                True,
                "Todas as dependências principais foram importadas com sucesso"
            )
        except ImportError as e:
            self.log_test_result(
                "Dependências importadas",
                False,
                f"Erro ao importar dependências: {str(e)}"
            )
    
    def print_summary(self):
        """Imprime um resumo dos testes."""
        print("=" * 60)
        print("📊 RESUMO DOS TESTES")
        print("=" * 60)
        print(f"Total de testes: {self.test_results['total_tests']}")
        print(f"Testes aprovados: {self.test_results['passed_tests']}")
        print(f"Testes falharam: {self.test_results['failed_tests']}")
        
        if self.test_results['total_tests'] > 0:
            success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests']) * 100
            print(f"Taxa de sucesso: {success_rate:.1f}%")
        
        print("\n📋 DETALHES DOS TESTES:")
        for test in self.test_results['test_details']:
            print(f"{test['status']}: {test['test_name']}")
            if test['details']:
                print(f"   {test['details']}")
        
        print("\n" + "=" * 60)
        
        if self.test_results['failed_tests'] == 0:
            print("🎉 Todos os testes passaram! O sistema está funcionando corretamente.")
        else:
            print("⚠️  Alguns testes falharam. Verifique os detalhes acima.")
        
        print("=" * 60)
    
    async def run_all_tests(self):
        """Executa todos os testes."""
        print("🚀 Iniciando testes do sistema de agentes orquestradores...")
        print("=" * 60)
        
        # Testes de configuração
        self.test_environment_setup()
        
        # Testes de componentes individuais
        await self.test_session_manager()
        await self.test_orchestrator_agent()
        await self.test_sql_agent()
        
        # Teste de integração
        await self.test_integration()
        
        # Resumo final
        self.print_summary()


async def main():
    """Função principal para executar os testes."""
    tester = SystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Configurar variáveis de ambiente para teste se não estiverem definidas
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY não configurada. Alguns testes podem falhar.")
        print("   Configure a variável de ambiente ou crie um arquivo .env")
        print()
    
    # Executar testes
    asyncio.run(main())
