📌 Índice

Descrição do Projeto

Pré-requisitos

Instalação

Explicação do Código

Importação das Bibliotecas

Configuração do Modelo e API

Gerenciamento de Histórico de Conversação

Uso de Prompt Templates

Gerenciamento de Memória do Chatbot

Pipeline de Execução

Como Contribuir

Glossário

📌 Descrição do Projeto 

Este projeto implementa um chatbot interativo usando a biblioteca LangChain integrada à API da Groq. Ele permite manter o contexto da conversa, aplicar otimizações no histórico de mensagens e estruturar fluxos conversacionais eficientes.

🔧 Pré-requisitos 

Antes de instalar o projeto, certifique-se de ter:

Python 3.8+

Um ambiente virtual configurado (recomendado)

API Key do Groq

Arquivo .env contendo a chave da API

🛠️ Instalação 

Clone este repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git

Acesse a pasta do projeto:

cd seu-repositorio

Crie um ambiente virtual e ative:

python -m venv venv
source venv/bin/activate  # Para Linux/macOS
venv\Scripts\activate  # Para Windows

Instale as dependências:

pip install -r requirements.txt

Crie um arquivo .env e adicione sua chave da API:

echo "GROQ_API_KEY=your_api_key" > .env

Execute o código para testar:

python main.py

📝 Explicação do Código 

1️⃣ Importação das Bibliotecas 

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

Aqui, são importadas bibliotecas essenciais para carregar variáveis de ambiente, interagir com a API do Groq e gerenciar o histórico de mensagens do chatbot.

2️⃣ Configuração do Modelo e API 

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

Essa parte do código carrega as credenciais da API a partir de um arquivo .env e inicializa o modelo de IA.

3️⃣ Gerenciamento de Histórico de Conversação 

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

Esse trecho cria um dicionário para armazenar o histórico de conversação de cada sessão de usuário, permitindo continuidade na interação.

4️⃣ Uso de Prompt Templates 

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Responda todas as perguntas com precisão."),
    MessagesPlaceholder(variable_name="messages")
])

Aqui, é definido um template de prompt para estruturar a entrada de mensagens do usuário e garantir respostas consistentes do chatbot.

5️⃣ Gerenciamento de Memória do Chatbot 

trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

Esse código define um limitador de mensagens para evitar que o histórico fique muito grande e ultrapasse os limites do modelo.

6️⃣ Pipeline de Execução 

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
response = chain.invoke({
    "messages": messages + [HumanMessage(content="Qual sorvete eu gosto?")],
    "language": "Português"
})

Aqui, é montado um pipeline para processar as mensagens, aplicar otimizações no histórico e formatar a entrada do modelo.

🤝 Como Contribuir 

Faça um fork do repositório

Crie uma branch para sua feature:

git checkout -b minha-feature

Faça suas alterações e commit:

git commit -m "Adicionei nova funcionalidade"

Envie suas alterações:

git push origin minha-feature

Abra um Pull Request no GitHub

📖 Glossário 

LangChain: Framework para construção de aplicações baseadas em LLMs (Modelos de Linguagem Extensa).

Groq API: Plataforma que fornece modelos de IA otimizados para processamento de linguagem natural.

Chatbot: Programa que interage automaticamente com usuários simulando uma conversa humana.

Prompt Template: Estrutura que define como as mensagens de entrada são organizadas antes de serem processadas pelo modelo.

Token: Unidade de texto processada pelo modelo, podendo ser palavras ou partes de palavras.

Pipeline: Cadeia de processamento onde os dados passam por várias etapas até a obtenção da resposta final.

📌 Siga as diretrizes para contribuir e ajude a melhorar este projeto! 🚀

1. Fork este repositório
2. Crie uma nova branch (`git checkout -b feature/sua-feature`)
3. Faça as alterações desejadas e commit (`git commit -m "feat: descrição da mudança"`)
4. Envie para o seu fork (`git push origin feature/sua-feature`)
5. Abra um Pull Request explicando suas mudanças

---
Agora o README está completo, explicando cada parte do código, incluindo instalação, execução e contribuições. 🚀
