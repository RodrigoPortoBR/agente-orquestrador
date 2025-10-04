/**
 * Integração com Lovable - Código JavaScript para ser usado no chat do Lovable
 * Este código deve ser incorporado no projeto Lovable para enviar mensagens via webhook
 */

// Configuração da integração
const WEBHOOK_CONFIG = {
    // URL do seu agente orquestrador (substitua pela URL real)
    webhookUrl: 'https://your-agent-api.com/webhook/lovable',
    
    // Timeout para requisições (em milissegundos)
    timeout: 30000,
    
    // Configurações de retry
    maxRetries: 3,
    retryDelay: 1000
};

// Classe para gerenciar a integração com o agente orquestrador
class AgentOrchestrator {
    constructor(config = WEBHOOK_CONFIG) {
        this.config = config;
        this.sessionId = this.generateSessionId();
        this.conversationHistory = [];
        this.isProcessing = false;
    }
    
    /**
     * Gera um ID único para a sessão
     */
    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    /**
     * Adiciona uma mensagem ao histórico local
     */
    addToHistory(role, content, metadata = null) {
        const message = {
            role: role,
            content: content,
            timestamp: new Date().toISOString(),
            metadata: metadata
        };
        
        this.conversationHistory.push(message);
        
        // Manter apenas as últimas 20 mensagens para evitar payload muito grande
        if (this.conversationHistory.length > 20) {
            this.conversationHistory = this.conversationHistory.slice(-20);
        }
    }
    
    /**
     * Envia mensagem para o agente orquestrador
     */
    async sendMessage(userMessage, userId = null) {
        if (this.isProcessing) {
            throw new Error('Já existe uma mensagem sendo processada. Aguarde.');
        }
        
        this.isProcessing = true;
        
        try {
            // Adicionar mensagem do usuário ao histórico
            this.addToHistory('user', userMessage);
            
            // Preparar payload
            const payload = {
                session_id: this.sessionId,
                user_message: userMessage,
                conversation_history: this.conversationHistory,
                timestamp: new Date().toISOString(),
                user_id: userId,
                metadata: {
                    source: 'lovable_chat',
                    user_agent: navigator.userAgent,
                    timestamp: Date.now()
                }
            };
            
            // Enviar requisição com retry
            const response = await this.sendWithRetry(payload);
            
            // Adicionar resposta do assistente ao histórico
            this.addToHistory('assistant', response.response, response.metadata);
            
            return response;
            
        } catch (error) {
            console.error('Erro ao enviar mensagem:', error);
            
            // Resposta de fallback em caso de erro
            const fallbackResponse = {
                response: 'Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente em alguns instantes.',
                session_id: this.sessionId,
                success: false,
                timestamp: new Date().toISOString(),
                metadata: {
                    error: error.message,
                    fallback: true
                }
            };
            
            this.addToHistory('assistant', fallbackResponse.response, fallbackResponse.metadata);
            return fallbackResponse;
            
        } finally {
            this.isProcessing = false;
        }
    }
    
    /**
     * Envia requisição com sistema de retry
     */
    async sendWithRetry(payload, attempt = 1) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);
            
            const response = await fetch(this.config.webhookUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            return data;
            
        } catch (error) {
            console.warn(`Tentativa ${attempt} falhou:`, error.message);
            
            if (attempt < this.config.maxRetries) {
                // Aguardar antes de tentar novamente
                await new Promise(resolve => 
                    setTimeout(resolve, this.config.retryDelay * attempt)
                );
                return this.sendWithRetry(payload, attempt + 1);
            }
            
            throw error;
        }
    }
    
    /**
     * Limpa o histórico da conversa
     */
    clearHistory() {
        this.conversationHistory = [];
        this.sessionId = this.generateSessionId();
    }
    
    /**
     * Obtém estatísticas da sessão
     */
    getSessionStats() {
        const userMessages = this.conversationHistory.filter(msg => msg.role === 'user').length;
        const assistantMessages = this.conversationHistory.filter(msg => msg.role === 'assistant').length;
        
        return {
            sessionId: this.sessionId,
            totalMessages: this.conversationHistory.length,
            userMessages: userMessages,
            assistantMessages: assistantMessages,
            isProcessing: this.isProcessing
        };
    }
}

// Classe para integração com interface do Lovable
class LovableChatIntegration {
    constructor() {
        this.orchestrator = new AgentOrchestrator();
        this.chatContainer = null;
        this.messageInput = null;
        this.sendButton = null;
        this.isInitialized = false;
    }
    
    /**
     * Inicializa a integração com o chat do Lovable
     */
    async initialize() {
        if (this.isInitialized) {
            return;
        }
        
        try {
            // Aguardar elementos do chat estarem disponíveis
            await this.waitForChatElements();
            
            // Configurar event listeners
            this.setupEventListeners();
            
            // Adicionar indicadores visuais
            this.addStatusIndicators();
            
            this.isInitialized = true;
            console.log('Integração com agente orquestrador inicializada com sucesso');
            
            // Enviar mensagem de boas-vindas (opcional)
            this.showWelcomeMessage();
            
        } catch (error) {
            console.error('Erro ao inicializar integração:', error);
        }
    }
    
    /**
     * Aguarda elementos do chat estarem disponíveis no DOM
     */
    async waitForChatElements() {
        const maxAttempts = 50;
        let attempts = 0;
        
        while (attempts < maxAttempts) {
            // Tentar encontrar elementos comuns de chat
            this.chatContainer = document.querySelector('.chat-container, .messages, .chat-messages, [class*="chat"]');
            this.messageInput = document.querySelector('input[type="text"], textarea, [placeholder*="message"], [placeholder*="pergunta"]');
            this.sendButton = document.querySelector('button[type="submit"], .send-button, [class*="send"]');
            
            if (this.messageInput) {
                break;
            }
            
            attempts++;
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        if (!this.messageInput) {
            throw new Error('Não foi possível encontrar elementos do chat');
        }
    }
    
    /**
     * Configura event listeners para interceptar mensagens
     */
    setupEventListeners() {
        // Interceptar envio de mensagens
        if (this.sendButton) {
            this.sendButton.addEventListener('click', (e) => this.handleMessageSend(e));
        }
        
        // Interceptar Enter no input
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                this.handleMessageSend(e);
            }
        });
        
        // Adicionar comando especial para limpar histórico
        this.messageInput.addEventListener('input', (e) => {
            if (e.target.value.toLowerCase() === '/clear') {
                this.orchestrator.clearHistory();
                e.target.value = '';
                this.showSystemMessage('Histórico da conversa limpo.');
            }
        });
    }
    
    /**
     * Manipula o envio de mensagens
     */
    async handleMessageSend(event) {
        event.preventDefault();
        event.stopPropagation();
        
        const message = this.messageInput.value.trim();
        if (!message) {
            return;
        }
        
        // Limpar input
        this.messageInput.value = '';
        
        // Mostrar indicador de carregamento
        this.showLoadingIndicator();
        
        try {
            // Enviar para o agente orquestrador
            const response = await this.orchestrator.sendMessage(message);
            
            // Exibir resposta
            this.displayResponse(response);
            
        } catch (error) {
            console.error('Erro ao processar mensagem:', error);
            this.showErrorMessage('Erro ao processar mensagem. Tente novamente.');
        } finally {
            this.hideLoadingIndicator();
        }
    }
    
    /**
     * Exibe a resposta do agente na interface
     */
    displayResponse(response) {
        // Criar elemento de mensagem
        const messageElement = document.createElement('div');
        messageElement.className = 'agent-response';
        messageElement.innerHTML = `
            <div class="message-content">
                <strong>Assistente:</strong> ${this.formatMessage(response.response)}
            </div>
            <div class="message-metadata">
                <small>
                    ${response.success ? '✅' : '❌'} 
                    ${new Date(response.timestamp).toLocaleTimeString()}
                    ${response.metadata?.intent ? `| ${response.metadata.intent}` : ''}
                </small>
            </div>
        `;
        
        // Adicionar ao container de chat
        if (this.chatContainer) {
            this.chatContainer.appendChild(messageElement);
            this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        }
    }
    
    /**
     * Formata mensagem para exibição
     */
    formatMessage(message) {
        // Converter quebras de linha para <br>
        return message.replace(/\n/g, '<br>');
    }
    
    /**
     * Mostra indicador de carregamento
     */
    showLoadingIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'agent-loading';
        indicator.innerHTML = '🤖 Processando...';
        indicator.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #007bff;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            z-index: 9999;
        `;
        
        document.body.appendChild(indicator);
    }
    
    /**
     * Remove indicador de carregamento
     */
    hideLoadingIndicator() {
        const indicator = document.getElementById('agent-loading');
        if (indicator) {
            indicator.remove();
        }
    }
    
    /**
     * Mostra mensagem de sistema
     */
    showSystemMessage(message) {
        const systemMsg = document.createElement('div');
        systemMsg.innerHTML = `<em>Sistema: ${message}</em>`;
        systemMsg.style.cssText = 'color: #666; font-style: italic; margin: 10px 0;';
        
        if (this.chatContainer) {
            this.chatContainer.appendChild(systemMsg);
        }
    }
    
    /**
     * Mostra mensagem de erro
     */
    showErrorMessage(message) {
        const errorMsg = document.createElement('div');
        errorMsg.innerHTML = `<strong style="color: red;">Erro: ${message}</strong>`;
        
        if (this.chatContainer) {
            this.chatContainer.appendChild(errorMsg);
        }
    }
    
    /**
     * Adiciona indicadores de status
     */
    addStatusIndicators() {
        const statusDiv = document.createElement('div');
        statusDiv.id = 'agent-status';
        statusDiv.innerHTML = '🤖 Agente conectado';
        statusDiv.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 12px;
            z-index: 9999;
        `;
        
        document.body.appendChild(statusDiv);
        
        // Remover após 3 segundos
        setTimeout(() => statusDiv.remove(), 3000);
    }
    
    /**
     * Mostra mensagem de boas-vindas
     */
    showWelcomeMessage() {
        setTimeout(() => {
            this.showSystemMessage('Agente orquestrador ativado! Digite suas perguntas normalmente.');
        }, 1000);
    }
}

// Inicializar integração quando a página carregar
if (typeof window !== 'undefined') {
    // Aguardar DOM estar pronto
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeIntegration);
    } else {
        initializeIntegration();
    }
}

async function initializeIntegration() {
    try {
        const integration = new LovableChatIntegration();
        await integration.initialize();
        
        // Tornar disponível globalmente para debug
        window.agentIntegration = integration;
        
    } catch (error) {
        console.error('Falha ao inicializar integração com agente:', error);
    }
}

// Exportar para uso em módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AgentOrchestrator,
        LovableChatIntegration
    };
}
