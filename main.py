# IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
# PERMITE CRIAR HISTORICO DE MENSAGENS
from langchain_community.chat_message_histories import ChatMessageHistory
# CRIA UMA CLASSE BASE PARA HISTÓRICO DE MENSAGENS
from langchain_core.chat_history import BaseChatMessageHistory
# PERMITE GERENCIAR O HISTÓRICO DE MENSAGENS
from langchain_core.runnables.history import RunnableWithMessageHistory
# PERMITE CRIAR PROMPTS / MENSAGENS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# MENSAGENS HUMANAS, DO SISTEMA E DO AI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
# PERMITE CRIAR FLUXOS DE EXECUÇÃO E REUTILIZAVEIS
from langchain_core.runnables import RunnablePassthrough
# FACILITA A EXTRAÇÃO DE VALORES DE DICIONÁRIOS
from operator import itemgetter


# CARREGAR AS VARIÁVEIS DE AMBIENTE DO ARQUIVO .ENV (PARA PROTEGER AS CREDENCIAIS)
load_dotenv(find_dotenv())

# OBTER A CHAVE DA API DO GROQ ARMAZENADA NO ARQUIVO .ENV
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# INICIALIZAR O MODELO DE AI UTILIZANDO A API DA GROQ
model = ChatGroq(
    model="gemma2-9b-it",
    groq_api_key=GROQ_API_KEY
)

# EXEMPLO 1
# CRIAR UM DICIONARIO PARA ARMAZENAR O HISTORICO DE MENSAGENS
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Recupera ou cria um historico de mensagens para uma determinada seção.
    Isso permite manter o contexto continuo para diferentes usuários e interações.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# CRIAR UM GERENCIADOR DE HISTÓRICO QUE CONECTA O MODELO AO ARMAZENAMENTO DE MENSAGENS
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# CONFIGURAÇÃO DA SESSÃO (IDENTIFICADOR ÚNICO PARA CADA CHAT/USUÁRIO)
config = {"configurable": {"session_id": "chat1"}}

# EXEMPLO DE INTERAÇÃO INICIAL DO USUÁRIO
response = with_message_history.invoke(
    [HumanMessage(content="Meu nome é Otávio e sou um filósofo")],
    config=config
)


# EXEMPLO 2
# CRIAÇÃO DE UM PROMPT TEMPLATE PARA ESTRUTURAR A ENTRADA DO MODELO
prompt = ChatPromptTemplate.from_messages
[
    ("system",
     "Vocé é um assistente útil. Responda todas as perguntas com precisão."),
    # PERMITIR ADICIONAR MENSAGENS DE FORMA DINÂMICA
    MessagesPlaceholder(variable_name="messages")
]

# CONECTAR O MODELO AO TEMPLATE DE PROMPT
chain = prompt | model  # USANDO LCEL PARA CONECTAR O PROMPT AO MODELO

# EXEMPLO DE INTERAÇÃO COM O MODELO DO TEMPLATE
response = chain.invoke(
    {"messages": [HumanMessage(content="Oi, o meu nome é Otávio!")]}
)

# GERENCIAMENTO DA MEMÓRIA DO CHATBOT
trimmer = trim_messages(
    max_tokens=45,  # DEFINE UM LIMITE MÁXIMO DE TOKENS PARA EVITAR ULTRAPASSAR O CONSUMO DE MEMÓRIA

    strategy="last",       # DEFINE A ESTRATEGIA DE CORTE PARA REMOVER MENSAGENS ANTIGAS
    token_counter=model,  # USA O MODELO PARA CONTAR OS TOKENS
    include_system=True,  # INCLUI AS MENSAGENS DO SISTEMA NO HISTÓRICO
    allow_partial=False,  # EVITA QUE AS MENSAGENS SEJAM CORTAS PARCIALMENTE
    start_on="human"      # COMEÇA A CONTAGEM DOS TOKENS COM A MENSAGEM HUMANA
)

# EXEMPLO DE HISTÓRICO DE MENSAGENS
messages = [
    SystemMessage(
        content="Você é um assistente. Responda todas as perguntas com precisão no idioma."),
    HumanMessage(content="Oi, meu nome é John Wick."),
    AIMessage(content="Oi John! Como posso te ajudar hoje?"),
    HumanMessage(content="Eu gosto de sorvete de doce de leite.")
]

# APLICAR O LIMITADOR DE MEMORIA AO HISTÓRICO
response = trimmer.invoke(messages)

# CRIANDO UM PIPELINE DE EXECUÇÃO PARA OTIMIZAR A PASSAGEM DE INFORMAÇÃO ENTRE OS COMPONENTES
chain = (
    RunnablePassthrough.assign(messages=itemgetter(
        "messages") | trimmer)  # APLICA OTIMIZAÇÃO DO HISTÓRICO
    | prompt  # PASSA A ENTRADA PELO TEMPLATE DE PROMPT
    | model  # PASSA A ENTRADA PELO MODELO
)

# EXEMPLO DE INTERAÇÃO UTILIZANDO O PIPELINE OTIMIZADO
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="Qual o sorvete que eu gosto?")]
    }
)

# EXIBIR A RESPOSTA FINAL DO MODELO
print("Resposta final do modelo: ", response.content)
