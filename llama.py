import sys
import os
import time
from dotenv import load_dotenv
import json
import logging
import subprocess
from logging.handlers import RotatingFileHandler
import threading
import websockets
import asyncio
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import requests
import numpy as np
import base64
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm

load_dotenv() 


venv_path = os.path.abspath("env")

if sys.prefix != venv_path:
    if os.name == 'nt': 
        venv_python = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_path, "bin", "python")

    os.execv(venv_python, [venv_python] + sys.argv)


logger = logging.getLogger(__name__)
log_file = 'llama_flask_app.log'
file_handler = RotatingFileHandler(
    log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

# CORS(app, resources={r"/run_llama": {"origins": os.getenv('ALLOWED_ORIGIN')}})
# CORS(app, resources={r"/run_llama": {"origins": "*"}})
# CORS(app, resources={r"/run_llama": {"origins": ["http://192.168.12.218:3000", "https://npc-rust-engine.onrender.com"]}})

project_id = os.getenv('PROJECT_ID')
project_secret = os.getenv('PROJECT_SECRET')
url = 'https://ipfs.infura.io:5001/api/v0/add'
auth = base64.b64encode(f"{project_id}:{project_secret}".encode()).decode()

headers = {
    "Authorization": f"Basic {auth}"
}

API_KEY = os.getenv('OLLAMA_API_KEY')
if not API_KEY:
    logger.error("OLLAMA_API_KEY environment variable not set")
    raise ValueError("OLLAMA_API_KEY must be set")
logger.debug("OLLAMA_API_KEY loaded successfully")

llama_lock = threading.Lock()
llama_process_active = False
current_process_id = None 


def require_api_key(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json

        logger.debug("Checking API key in request...")
        if not data:
            logger.warning("No JSON data received in the request")
            return jsonify({"error": "API key is required"}), 400
        if 'api_key' not in data:
            logger.warning("API key missing in request")
            return jsonify({"error": "API key is required"}), 400
        if data['api_key'] != API_KEY:
            logger.warning("Invalid API key received")  
            return jsonify({"error": "Invalid API key"}), 401
        logger.debug("API key validated successfully")
        return f(*args, **kwargs)
    return decorated_function

def delete_json_file(json_file_path):
    retries = 5  
    while retries > 0:
        try:
            if os.path.exists(json_file_path):
                os.remove(json_file_path)
                logger.info(f"Archivo {json_file_path} borrado exitosamente.")
            else:
                logger.info(f"El archivo {json_file_path} no existe.")
            break  
        except PermissionError as e:
            logger.warning(f"Error al borrar el archivo {json_file_path}: {e}")
            retries -= 1
            time.sleep(1) 
            continue
    if retries == 0:
        logger.error(f"No se pudo borrar el archivo {json_file_path} después de varios intentos.")

@app.route('/run_llama', methods=['POST'])
@require_api_key 
def run_llama():
    try:
        global llama_process_active, current_process_id
        if llama_lock.locked():
            return jsonify({"error": "llama-cli is already running"}), 429 
        data = request.json
        prompt = data.get('prompt', '').strip()
        options = data.get('options', {})
        model = data.get('model', "E:\\dev\\llama.cpp\\models\\8B\\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf").strip()
        images = data.get('images', '')
        metadata_uri = data.get('metadata_uri', '').strip()
        locale = data.get('locale', '').strip()
        eleccion = data.get('eleccion', 0)
        comentario_perfil = data.get('comentario_perfil', 0)
        comentario_pub = data.get('comentario_pub', 0)
        perfil_id = data.get('perfil_id', 0)
        coleccion_id = data.get('coleccion_id', 0)
        pagina = data.get('pagina', 0)
        clave = data.get('escena', '').strip()
        npc = data.get('id', '').strip()  
    
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        logger.info(f"Received prompt: {prompt}")

        
        threading.Thread(target=background_llama, args=(prompt, options, model, images,
metadata_uri,
locale,
eleccion,
comentario_perfil,
comentario_pub,
perfil_id,
coleccion_id,
pagina,
clave,
npc)).start()
        return jsonify({"message": "Solicitud recibida, procesando en segundo plano"}), 200


    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    

def convertir_cadena_a_lista(cadena):
    try:
        cadena = cadena.replace("null", "0.0")
        lista = json.loads(cadena.replace("'", "\""))
        return lista
    except json.JSONDecodeError as e:
        print(f"Error al convertir la cadena a lista: {e}")
        return []
    
def cargar_embeddings(campo1):
    with open('llama_output.json', 'r') as f:
        data = json.load(f)
    embeddings = convertir_cadena_a_lista(data[campo1])
    embeddings = np.array(embeddings, dtype=np.float64)
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e9, neginf=-1e9)
    return embeddings

def calcular_normas(embeddings):
    normas = np.linalg.norm(embeddings, axis=1)
    return normas

def calcular_ortogonalidad(embeddings):
    num_tokens = embeddings.shape[0]
    productos_punto = np.zeros((num_tokens, num_tokens))

    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            producto_punto = np.dot(embeddings[i], embeddings[j])
            productos_punto[i, j] = producto_punto
            productos_punto[j, i] = producto_punto

    return productos_punto

def graficar_normas(campo, normas, title):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='black') 
    ax.bar(range(len(normas)), normas, color='#CC04FD')         
    ax.set_facecolor('black')                               
    plt.xlabel('Token', color='white')
    plt.ylabel('Norm', color='white')
    plt.title(f'Embedding Norms for {title}', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.grid(False) 
    plt.savefig(f"{campo}_norms.png", bbox_inches='tight', facecolor='black')  
    plt.clf()

def graficar_ortogonalidad(campo, productos_punto, title):
    plt.figure(figsize=(10, 8), facecolor='black')
    plt.imshow(productos_punto, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Dot Product')
    plt.title(f'Orthogonality between Embeddings for {title}', color='white')
    plt.xlabel('Token', color='white')
    plt.ylabel('Token', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.grid(False)  
    plt.savefig(f"{campo}_orthogonality.png", bbox_inches='tight', facecolor='black')
    plt.clf()

def analizar_embeddings(campo, titulo):
    embeddings = cargar_embeddings(campo)

    if embeddings.shape[0] == 1:
        print(f"Only one vector for {campo}. Skipping orthogonality plot.")
        normas = calcular_normas(embeddings)
        graficar_normas(campo, normas, titulo)
    else:
        normas = calcular_normas(embeddings)
        productos_punto = calcular_ortogonalidad(embeddings)
        graficar_normas(campo, normas, titulo)
        graficar_ortogonalidad(campo, productos_punto, titulo)


def generar_graficos(campo1, campo2, numero_1, numero_2, titulo):
    with open('llama_output.json', 'r') as f:
        data = json.load(f)
    inputs_procesados = convertir_cadena_a_lista(data[campo1])
    outputs_procesados = convertir_cadena_a_lista(data[campo2])

    inputs = np.array(inputs_procesados, dtype=np.float64)
    outputs = np.array(outputs_procesados, dtype=np.float64)
    inputs = np.nan_to_num(inputs, nan=0.0, posinf=1e9, neginf=-1e9)
    outputs = np.nan_to_num(outputs, nan=0.0, posinf=1e9, neginf=-1e9)

    inputs = np.clip(inputs, -1e9, 1e9)
    outputs = np.clip(outputs, -1e9, 1e9)

    min_tokens = min(inputs.shape[0], outputs.shape[0])
    inputs = inputs[:min_tokens]
    outputs = outputs[:min_tokens]

    embeddings = np.vstack((inputs, outputs))

    labels = ['first'] * len(inputs) + ['last'] * len(outputs)

    assert len(labels) == embeddings.shape[0], "La longitud de las etiquetas no coincide con los embeddings"

    pca = PCA(n_components=2)
    reduced_embeddings_pca = pca.fit_transform(embeddings)

    num_samples = embeddings.shape[0]
    perplexity = min(30, num_samples - 1)  

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    reduced_embeddings_tsne = tsne.fit_transform(embeddings)

    num_tokens = len(inputs)
    palette = sns.color_palette("hsv", num_tokens)

    plt.figure(figsize=(14, 7), facecolor='black')
    ax1 = plt.subplot(1, 2, 1, facecolor='black')
    for label in np.unique(labels):
        indices = np.where(np.array(labels) == label)
        plt.scatter(reduced_embeddings_pca[indices, 0], reduced_embeddings_pca[indices, 1], color=['#CC04FD' if label == 'first' else '#23DBAA'], alpha=0.7)
    
    ax1.set_xticks([])
    ax1.set_yticks([]) 
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')
    
    plt.legend(titulo, loc="lower right", fontsize=10)

    ax2 = plt.subplot(1, 2, 2, facecolor='black')
    for label in np.unique(labels):
        indices = np.where(np.array(labels) == label)
        plt.scatter(reduced_embeddings_tsne[indices, 0], reduced_embeddings_tsne[indices, 1], color=['#CC04FD' if label == 'first' else '#23DBAA'], alpha=0.7)
    
    ax2.set_xticks([])
    ax2.set_yticks([]) 
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

    plt.legend(titulo,  loc="lower right", fontsize=10)

    plt.savefig(numero_1, bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(14, 7), facecolor='black')
    ax1 = plt.subplot(1, 2, 1, facecolor='black')
    for i in range(num_tokens):
        plt.scatter(reduced_embeddings_pca[i, 0], reduced_embeddings_pca[i, 1], color=palette[i], alpha=0.7)
        plt.scatter(reduced_embeddings_pca[num_tokens + i, 0], reduced_embeddings_pca[num_tokens + i, 1], color=palette[i], alpha=0.7)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['right'].set_color('white')

    ax2 = plt.subplot(1, 2, 2, facecolor='black')
    for i in range(num_tokens):
        plt.scatter(reduced_embeddings_tsne[i, 0], reduced_embeddings_tsne[i, 1], color=palette[i], alpha=0.7)
        plt.scatter(reduced_embeddings_tsne[num_tokens + i, 0], reduced_embeddings_tsne[num_tokens + i, 1], alpha=0.7)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

    plt.savefig(numero_2, bbox_inches='tight')
    plt.clf()


def generar_prompt():
    with open('llama_output.json', 'r') as f:
        data = json.load(f)
    prompt_crudo = convertir_cadena_a_lista(data["inputs_prompt"])
    embedding_global = convertir_cadena_a_lista(data["outputs_prompt"])
    res = convertir_cadena_a_lista(data["res_prompt"])

    prompt_crudo = np.array(prompt_crudo, dtype=np.float64)
    embedding_global = np.array(embedding_global, dtype=np.float64)
    res = np.array(res, dtype=np.float64)

    prompt_crudo = np.nan_to_num(prompt_crudo, nan=0.0, posinf=1e9, neginf=-1e9)
    embedding_global = np.nan_to_num(embedding_global, nan=0.0, posinf=1e9, neginf=-1e9)
    res = np.nan_to_num(res, nan=0.0, posinf=1e9, neginf=-1e9)

    scaler = StandardScaler()

    prompt_crudo_normalizado = scaler.fit_transform(prompt_crudo)
    embedding_global_normalizado = scaler.transform(embedding_global.reshape(1, -1))
    res_normalizado = scaler.transform(res.reshape(1, -1))

    todos_los_embeddings = np.vstack([prompt_crudo_normalizado, embedding_global_normalizado, res_normalizado])

    pca = PCA(n_components=2)
    embedding_2d_pca = pca.fit_transform(todos_los_embeddings)

    tsne = TSNE(n_components=2, perplexity=3, n_iter=300, random_state=42)
    embedding_2d_tsne = tsne.fit_transform(todos_los_embeddings)

    plt.style.use('dark_background')

    plt.figure(figsize=(10, 6))

    plt.scatter(embedding_2d_pca[:-2, 0], embedding_2d_pca[:-2, 1], c='#CC04FD', label='Token Embeddings', marker='o') 

    plt.scatter(embedding_2d_pca[-2, 0], embedding_2d_pca[-2, 1], c='#23DBAA', label='Global Embedding', marker='o', s=100)

    plt.scatter(embedding_2d_pca[-1, 0], embedding_2d_pca[-1, 1], c='#F6FC8D', label='Logits', marker='o', s=100)

    for i in range(len(prompt_crudo_normalizado)):
        plt.plot([embedding_2d_pca[i, 0], embedding_2d_pca[-2, 0]],
                [embedding_2d_pca[i, 1], embedding_2d_pca[-2, 1]],
                'w--', linewidth=0.6)  

    plt.legend() 
    plt.grid(False)  
    plt.gca().spines['top'].set_color('white') 
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.savefig("Prompt_PCA", bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(10, 6))

    plt.scatter(embedding_2d_tsne[:-2, 0], embedding_2d_tsne[:-2, 1], c='#CC04FD', label='Token Embeddings', marker='o')

    plt.scatter(embedding_2d_tsne[-2, 0], embedding_2d_tsne[-2, 1], c='#23DBAA', label='Global Embedding', marker='o', s=100)

    plt.scatter(embedding_2d_tsne[-1, 0], embedding_2d_tsne[-1, 1], c='#F6FC8D', label='Logits', marker='o', s=100)

    for i in range(len(prompt_crudo_normalizado)):
        plt.plot([embedding_2d_tsne[i, 0], embedding_2d_tsne[-2, 0]],
                [embedding_2d_tsne[i, 1], embedding_2d_tsne[-2, 1]],
                'w--', linewidth=0.6) 

    plt.legend()
    plt.grid(False)
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.savefig("Prompt_TSNE", bbox_inches='tight')
    plt.clf()



def subir_a_ipfs(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, headers=headers, files=files)
            if response.status_code == 200:
                json_response = response.json()
                cid = json_response["Hash"]  
                return "ipfs://" + cid
            else:
                print(f"Error: {response.text}")
                return None
    else:
        print(f"Archivo no encontrado en la ruta: {file_path}")
        return None
    
def subir_json_a_ipfs(cadena):
    json_data = json.dumps(cadena)
    files = {'file': ('metadata.json', json_data, 'application/json')}
    
    response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        json_response = response.json()
        cid = json_response["Hash"]  
        return "ipfs://" + cid
    else:
        print(f"Error: {response.text}")
        return None

def subir_cadena_a_ipfs(cadena):
    files = {'file': ('cadena.txt', cadena, 'text/plain')}
    
    response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        json_response = response.json()
        cid = json_response["Hash"]  
        return "ipfs://" + cid
    else:
        print(f"Error: {response.text}")
        return None


def eliminar_graficos(files):
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Archivo {file} eliminado exitosamente.")
            else:
                print(f"El archivo {file} no existe.")
        except Exception as e:
            print(f"Error al intentar eliminar el archivo {file}: {e}")
    

def background_llama(prompt, options, model, images,
metadata_uri,
locale,
eleccion,
comentario_perfil,
comentario_pub,
perfil_id,
coleccion_id,
pagina,
clave,
npc):
    global llama_process_active, current_process_id
    try:
        with llama_lock:
            llama_process_active = True
            current_process_id = 1
            ipfs_hashes = []
            json_result = ejecutar_llama(prompt, options, model)
            generar_graficos("inputs_respuesta", "outputs_respuesta", "Respuesta_Todo", "Respuesta_Indi", ["Raw Token Embeddings", "Transformed Token Embeddings"])
            generar_graficos("inputs_respuesta", "res_respuesta", "Respuesta_Todo_Res", "Respuesta_Indi_Res", ["Raw Token Embeddings", "Token Logits"])
            generar_prompt()
            campos_a_analizar = ['inputs_prompt', 'outputs_prompt', 'res_prompt', 'inputs_respuesta', 'outputs_respuesta', "res_respuesta"]
            titulos = ["Raw Embeddings for Prompt Tokens", "Transformed Embedding for Prompt", "Logits for Prompt", "Raw Embeddings for Response Tokens", "Transformed Embedding for Response Tokens", "Logits for Response"]

            for campo, titulo in zip(campos_a_analizar, titulos):
                analizar_embeddings(campo, titulo)

            for image_path in ["Respuesta_Todo", "Respuesta_Indi", "Respuesta_Todo_Res", "Respuesta_Indi_Res", "Prompt_PCA", "Prompt_TSNE", 'inputs_prompt_norms', 'inputs_prompt_orthogonality', 'outputs_prompt_norms', 'res_prompt_norms', 'inputs_respuesta_norms', 'outputs_respuesta_norms', "res_respuesta_norms",  'inputs_respuesta_orthogonality', 'outputs_respuesta_orthogonality', "res_respuesta_orthogonality"]:
                cid = subir_a_ipfs(image_path)
                if cid:
                    ipfs_hashes.append(cid)

            eliminar_graficos(["Respuesta_Todo", "Respuesta_Indi", "Respuesta_Todo_Res", "Respuesta_Indi_Res", "Prompt_PCA", "Prompt_TSNE", 'inputs_prompt_norms', 'inputs_prompt_orthogonality', 'outputs_prompt_norms', 'res_prompt_norms', 'inputs_respuesta_norms', 'outputs_respuesta_norms', "res_respuesta_norms",  'inputs_respuesta_orthogonality', 'outputs_respuesta_orthogonality', "res_respuesta_orthogonality"])

            inputs_prompt = subir_cadena_a_ipfs(json_result.get("inputs_prompt"))
            outputs_prompt = subir_cadena_a_ipfs(json_result.get("outputs_prompt"))
            res_prompt = subir_cadena_a_ipfs(json_result.get("res_prompt"))
            inputs_respuesta = subir_cadena_a_ipfs(json_result.get("inputs_respuesta"))
            outputs_respuesta = subir_cadena_a_ipfs(json_result.get("outputs_respuesta"))
            res_respuesta = subir_cadena_a_ipfs(json_result.get("res_respuesta"))
            token_means_respuesta = subir_cadena_a_ipfs(json_result.get("token_means_respuesta"))
            k_means_respuesta = subir_cadena_a_ipfs(json_result.get("k_means_respuesta"))
            v_means_respuesta = subir_cadena_a_ipfs(json_result.get("v_means_respuesta"))
            value_std_devs_respuesta  =  subir_cadena_a_ipfs(json_result.get("value_std_devs_respuesta"))
            value_maxs_respuesta = subir_cadena_a_ipfs(json_result.get("value_maxs_respuesta"))
            value_mins_respuesta = subir_cadena_a_ipfs(json_result.get("value_mins_respuesta"))
            key_std_devs_respuesta = subir_cadena_a_ipfs(json_result.get("key_std_devs_respuesta"))
            key_maxs_respuesta = subir_cadena_a_ipfs(json_result.get("key_maxs_respuesta"))
            key_mins_respuesta = subir_cadena_a_ipfs(json_result.get("key_mins_respuesta"))
            ffn_out_std_devs = subir_cadena_a_ipfs(json_result.get("ffn_out_std_devs"))
            ffn_out_maxs = subir_cadena_a_ipfs(json_result.get("ffn_out_maxs"))
            ffn_out_mins = subir_cadena_a_ipfs(json_result.get("ffn_out_mins"))
            ffn_out_means = subir_cadena_a_ipfs(json_result.get("ffn_out_means"))
            ffn_inp_std_devs = subir_cadena_a_ipfs(json_result.get("ffn_inp_std_devs"))
            ffn_inp_maxs = subir_cadena_a_ipfs(json_result.get("ffn_inp_maxs"))
            ffn_inp_mins = subir_cadena_a_ipfs(json_result.get("ffn_inp_mins"))
            ffn_inp_means = subir_cadena_a_ipfs(json_result.get("ffn_inp_means"))

            cadena = subir_json_a_ipfs({
                    "images": images,
                    "metadata_uri": metadata_uri,
                    "locale": locale,
                    "eleccion": eleccion,
                    "comentario_perfil": comentario_perfil,
                    "comentario_pub": comentario_pub,
                    "perfil_id": perfil_id,
                    "coleccion_id": coleccion_id,
                    "pagina": pagina,
                    "prompt": prompt,
                    "options": options,
                    "version": 1,
                    "mensaje": {
                        "inputs_prompt": inputs_prompt,
                        "outputs_prompt": outputs_prompt,
                        "res_prompt": res_prompt,
                        "inputs_respuesta":inputs_respuesta ,
                        "outputs_respuesta":outputs_respuesta ,
                        "res_respuesta":res_respuesta ,
                        "token_means_respuesta":token_means_respuesta ,
                        "k_means_respuesta":k_means_respuesta ,
                        "v_means_respuesta": v_means_respuesta,
                        "value_std_devs_respuesta": value_std_devs_respuesta,
                        "value_maxs_respuesta":value_maxs_respuesta ,
                        "value_mins_respuesta": value_mins_respuesta,
                        "key_std_devs_respuesta": key_std_devs_respuesta,
                        "key_maxs_respuesta":key_maxs_respuesta ,
                        "key_mins_respuesta": key_mins_respuesta,
                        "ffn_out_std_devs": ffn_out_std_devs,
                        "ffn_out_maxs": ffn_out_maxs,
                        "ffn_out_mins": ffn_out_mins,
                        "ffn_out_means":ffn_out_means ,
                        "ffn_inp_std_devs":ffn_inp_std_devs ,
                        "ffn_inp_maxs":ffn_inp_maxs ,
                        "ffn_inp_mins": ffn_inp_mins,
                        "ffn_inp_means": ffn_inp_means,
                        "output": json_result.get("output"),
                        "input_tokens": json_result.get("input_tokens"),
                        "output_tokens":  json_result.get("output_tokens"),
                    },
                    "hashes": ipfs_hashes
                })
            
            resultado_json = {
                "response": json_result.get("output"),
                "json": cadena
                }
            
            final_json = {
                    "clave": clave,
                    "tipo": "llamaContenido",
                    "npc": npc,
                    "json": resultado_json 
                }
            

            if json:
                logger.info("Llama completado. Enviando resultado al cliente.")
                asyncio.run(enviar_respuesta_websocket(f"wss://npc-rust-engine.onrender.com?key={os.getenv('RENDER_API_KEY')}", final_json))
                # asyncio.run(enviar_respuesta_websocket(f"ws://192.168.12.218:8080?key={os.getenv('RENDER_API_KEY')}", final_json))
            else:
                logger.error("Error durante la ejecución de llama-cli.")

    except Exception as e:
        logger.error(f"Error inesperado en background_llama: {e}")
    finally:
        llama_process_active = False
        current_process_id = None
        delete_json_file("E:\\dev\\llama.cpp\\llama_output.json")

def ejecutar_llama(prompt, options, model):
    try:

        command = [
            "E:\\dev\\llama.cpp\\build\\bin\\Release\\llama-cli.exe",
            "-m", "E:\\dev\\llama.cpp\\models\\8B\\" + model,
            "-n", str(options.get('num_tokens', 200)),
            "--n-gpu-layers", str(options.get('num_gpu', 18)),
            # "--batch-size", str(128),
            "--ctx-size", str(8192),
            # "--seed", str(options.get('seed', -1)),
            # "--threads", str(options.get('num_thread', 8)),
            # "--predict", str(options.get('num_ctx', 100)),
            # "--temp", str(options.get('temperature', 0.8)),
            # "--repeat-penalty", str(options.get('repeat_penalty', 1.2)),
            # "--repeat-last-n", str(options.get('repeat_last_n', 33)),
            # "--top-k", str(options.get('top_k', 20)),
            # "--top-p", str(options.get('top_p', 0.9)),
            # "--min-p", str(options.get('min_p', 0.0)),
            # "--tfs", str(options.get('tfs_z', 0.5)),
            # "--typical", str(options.get('typical_p', 0.7)),
            # "--mirostat", str(options.get('mirostat', 1)),
            # "--mirostat-lr", str(options.get('mirostat_eta', 1)),
            # "--mirostat-ent", str(options.get('mirostat_tau', 1)),
            # "--keep", str(options.get('num_keep', 5)),
            "--color",
            "-p", prompt
        ]

        llama_output_path = "E:\\dev\\llama.cpp\\llama_output.json"
        logger.info("Ejecutando llama-cli...")

        exit_code = subprocess.call(command)

        if exit_code != 0:
            logger.warning(f"llama-cli terminó con código de salida {exit_code}")

        time.sleep(60)

        if not os.path.exists(llama_output_path):
            logger.error("Output file not generated")
            return None

        with open(llama_output_path, 'r', encoding='utf-8') as json_file:
            llama_output = json.load(json_file)
            return llama_output

    except Exception as e:
        logger.error(f"Unexpected error during llama execution: {e}", exc_info=True)
        return None


        
async def enviar_respuesta_websocket(url, output):
    """Conecta al cliente por WebSocket y envía la respuesta."""
    try:
        headers   = {
        "origin": "https://glorious-eft-deeply.ngrok-free.app"
    }
        async with websockets.connect(url, extra_headers=headers) as websocket:
            await websocket.send(json.dumps(output))
            logger.info(f"Resultado enviado a través de WebSocket a {url}")
    except Exception as e:
        logger.error(f"Error enviando resultado por WebSocket: {e}")
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





