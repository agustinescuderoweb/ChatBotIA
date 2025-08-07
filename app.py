from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import os
import openai
import faiss
import numpy as np
import tiktoken
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Inicializar Flask
app = Flask(__name__)
CORS(app)  # Permite CORS globalmente

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuración y constantes
EMBED_MODEL = "text-embedding-3-small"
DOCUMENT_ID = "1zp82IiUQyW75BTd4MgwetmLtQQN_GREdetQRYFfd3Cg"   # Cambia por tu ID real
UPDATE_INTERVAL = 5 * 60  # segundos

# Variables globales para índice y chunks
index = None
chunks = []
last_update = 0

# Función para agregar headers CORS a cada respuesta
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

# Función para extraer texto desde Google Docs
def extract_text_from_gdoc(document_id):
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
    SERVICE_ACCOUNT_FILE = 'CREDENTIALS_JSON'  # Tu archivo de credenciales JSON
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('docs', 'v1', credentials=creds)
    doc = service.documents().get(documentId=document_id).execute()
    text = ""
    for element in doc.get('body', {}).get('content', []):
        paragraph = element.get('paragraph')
        if paragraph:
            for el in paragraph.get('elements', []):
                text_run = el.get('textRun')
                if text_run:
                    text += text_run.get('content')
    return text

# Función para dividir texto en chunks por tokens
def chunk_text(text, max_tokens=500, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    paragraphs = text.split('\n')
    chunks, current_chunk = [], ""
    current_tokens = 0
    for para in paragraphs:
        tokens = len(enc.encode(para))
        if current_tokens + tokens <= max_tokens:
            current_chunk += para + "\n"
            current_tokens += tokens
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
            current_tokens = tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Obtener embedding con OpenAI
def get_embedding(text):
    res = openai.embeddings.create(input=[text], model=EMBED_MODEL)
    return res.data[0].embedding

# Crear índice vectorial FAISS
def create_vector_store(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

# Buscar chunks similares para contexto
def search_similar_chunks(query, index, chunks, top_k=3):
    query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

# Generar respuesta usando contexto y OpenAI chat completions
def ask_question_with_context(query, index, chunks):
    relevant_chunks = search_similar_chunks(query, index, chunks)
    context = "\n".join(relevant_chunks)

    prompt = f"""
Sos un asistente comercial amable y cercano que ayuda a clientes a entender los servicios web. Respondé la pregunta usando el contexto, respondé incluyendo emojis cuando sea apropiado, con un lenguaje simple y amigable, y siempre invitando a continuar la conversación.

Contexto:
{context}

Pregunta del usuario: {query}
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Sos un asistente comercial entrenado para responder preguntas de forma clara, resumida y convincente, usando información de un documento."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content

# Verificar si es necesario actualizar el índice
def needs_update():
    global last_update
    return (time.time() - last_update) > UPDATE_INTERVAL

# Actualizar índice con nuevo contenido de Google Docs
def update_index():
    global index, chunks, last_update
    print("Actualizando índice con documento de Google Docs...")
    texto = extract_text_from_gdoc(DOCUMENT_ID)
    chunks = chunk_text(texto)
    index = create_vector_store(chunks)
    last_update = time.time()
    print("Índice actualizado.")

# Ruta principal
@app.route('/')
def index_page():
    return render_template("index.html")

# Ruta para chatbot, con manejo de POST y OPTIONS (preflight CORS)
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        # Respuesta para preflight CORS
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    global index, chunks

    if index is None or needs_update():
        try:
            update_index()
        except Exception as e:
            return jsonify({"reply": f"Error al actualizar el documento: {str(e)}"})

    data = request.get_json()
    pregunta = data.get("message", "")
    try:
        respuesta = ask_question_with_context(pregunta, index, chunks)
        return jsonify({"reply": respuesta})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == "__main__":
    print("Iniciando servidor...")
    app.run(debug=True)


