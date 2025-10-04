# Guia de Integração com Lovable

Este documento fornece instruções detalhadas para integrar o sistema de agentes orquestradores com um chat criado no Lovable.

## Visão Geral da Integração

O sistema funciona da seguinte forma:

1. **Usuário** digita uma mensagem no chat do Lovable
2. **JavaScript de integração** intercepta a mensagem e envia via webhook
3. **Agente Orquestrador** analisa a intenção e processa a solicitação
4. **Agente SQL** (se necessário) executa consultas no banco de dados
5. **Resposta** é formatada em linguagem natural e retornada ao chat

## Pré-requisitos

### 1. Deploy do Sistema de Agentes

Primeiro, você precisa fazer o deploy da API do sistema de agentes:

```bash
# 1. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas configurações

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar a aplicação
python main.py
```

### 2. Configuração das Variáveis de Ambiente

Edite o arquivo `.env` com suas configurações:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Database Configuration (opcional para demonstração)
DATABASE_URL=postgresql://user:password@host:port/database

# Redis Configuration (opcional, usa memória se não configurado)
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
WEBHOOK_SECRET=your-webhook-secret-here
```

### 3. Deploy da API

Recomendamos usar uma das seguintes plataformas:

- **Railway**: Deploy automático via GitHub
- **Render**: Deploy gratuito com limitações
- **Heroku**: Deploy tradicional
- **Vercel**: Para aplicações serverless

## Integração no Lovable

### Passo 1: Criar o Chat Base

No Lovable, crie um chat básico com os seguintes elementos:

```javascript
// Exemplo de prompt para o Lovable:
"Crie um chat interface com:
- Input de texto para mensagens
- Área de exibição de mensagens
- Botão de envio
- Histórico de conversa
- Design moderno e responsivo"
```

### Passo 2: Adicionar o Código de Integração

Copie o conteúdo do arquivo `lovable_integration.js` e adicione ao seu projeto Lovable.

**No chat do Lovable, digite:**

```
Adicione este código JavaScript ao projeto:

[Cole aqui o conteúdo do arquivo lovable_integration.js]

Configure a URL do webhook para: https://sua-api-url.com/webhook/lovable
```

### Passo 3: Configurar a URL do Webhook

No código JavaScript, atualize a configuração:

```javascript
const WEBHOOK_CONFIG = {
    // Substitua pela URL real da sua API
    webhookUrl: 'https://sua-api-deployada.com/webhook/lovable',
    timeout: 30000,
    maxRetries: 3,
    retryDelay: 1000
};
```

### Passo 4: Personalizar a Interface (Opcional)

Você pode personalizar a aparência das mensagens do agente:

```css
.agent-response {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.message-content {
    margin-bottom: 8px;
}

.message-metadata {
    color: #666;
    font-size: 0.85em;
}
```

## Testando a Integração

### 1. Teste Básico

Após a integração, teste com mensagens simples:

- "Olá"
- "Como você funciona?"
- "Preciso de ajuda"

### 2. Teste de Consultas SQL

Se você configurou um banco de dados, teste consultas como:

- "Quantos usuários temos?"
- "Mostre as vendas do último mês"
- "Qual produto mais vendido?"

### 3. Verificar Logs

Monitore os logs da API para verificar se as mensagens estão sendo recebidas:

```bash
# Se executando localmente
tail -f logs/app.log

# Se usando Railway/Render, verifique os logs na plataforma
```

## Estrutura do Payload

O sistema envia os seguintes dados via webhook:

```json
{
  "session_id": "session_1234567890_abc123",
  "user_message": "Quantos usuários temos cadastrados?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Olá",
      "timestamp": "2025-10-03T20:30:00Z",
      "metadata": null
    },
    {
      "role": "assistant", 
      "content": "Olá! Como posso ajudá-lo?",
      "timestamp": "2025-10-03T20:30:01Z",
      "metadata": {"intent": "general_chat"}
    }
  ],
  "timestamp": "2025-10-03T20:31:00Z",
  "user_id": null,
  "metadata": {
    "source": "lovable_chat",
    "user_agent": "Mozilla/5.0...",
    "timestamp": 1696367460000
  }
}
```

## Resposta da API

A API retorna uma resposta no formato:

```json
{
  "response": "Atualmente temos 1.247 usuários cadastrados no sistema.",
  "session_id": "session_1234567890_abc123",
  "timestamp": "2025-10-03T20:31:02Z",
  "success": true,
  "metadata": {
    "intent": "sql_query",
    "confidence": 0.95,
    "execution_time": 0.234
  }
}
```

## Comandos Especiais

O sistema suporta alguns comandos especiais:

- `/clear` - Limpa o histórico da conversa
- `/stats` - Mostra estatísticas da sessão
- `/help` - Exibe ajuda sobre o sistema

## Solução de Problemas

### Problema: Mensagens não são enviadas

**Possíveis causas:**
- URL do webhook incorreta
- API não está rodando
- Problemas de CORS

**Solução:**
1. Verifique se a API está acessível
2. Teste o endpoint `/health` da API
3. Verifique os logs do navegador (F12)

### Problema: Respostas demoram muito

**Possíveis causas:**
- Consultas SQL complexas
- Timeout da OpenAI API
- Problemas de rede

**Solução:**
1. Aumente o timeout na configuração
2. Otimize consultas SQL
3. Verifique a configuração da OpenAI API

### Problema: Erros de validação

**Possíveis causas:**
- Formato do payload incorreto
- Campos obrigatórios ausentes

**Solução:**
1. Verifique o formato do JSON enviado
2. Consulte os logs da API para detalhes do erro

## Monitoramento

### Endpoints de Monitoramento

- `GET /health` - Status da aplicação
- `GET /stats` - Estatísticas de uso
- `GET /session/{session_id}` - Informações da sessão

### Métricas Importantes

- Taxa de sucesso das requisições
- Tempo médio de resposta
- Número de sessões ativas
- Erros por tipo

## Segurança

### Recomendações

1. **Use HTTPS** em produção
2. **Configure CORS** adequadamente
3. **Implemente rate limiting** se necessário
4. **Monitore logs** para atividades suspeitas
5. **Use secrets** para chaves de API

### Validação de Webhook

Para maior segurança, você pode implementar validação de webhook:

```javascript
// No código de integração
const webhookSecret = 'your-webhook-secret';
const signature = await crypto.subtle.digest('SHA-256', 
  new TextEncoder().encode(JSON.stringify(payload) + webhookSecret)
);
```

## Próximos Passos

1. **Teste** a integração completamente
2. **Monitore** o desempenho em produção
3. **Colete feedback** dos usuários
4. **Otimize** baseado no uso real
5. **Expanda** funcionalidades conforme necessário

## Suporte

Para problemas ou dúvidas:

1. Verifique os logs da aplicação
2. Teste os endpoints individualmente
3. Consulte a documentação da API
4. Verifique a configuração das variáveis de ambiente
