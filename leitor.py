from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import re
from rapidfuzz import fuzz
import unicodedata

app = Flask(__name__)
reader = easyocr.Reader(['pt'])

COMBUSTIVEIS = ["gasolina", "etanol", "diesel", "gnv"]

def normalizar_texto(texto):
    """
    Remove acentos e pontuação, e converte para minúsculas.
    """
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-zA-Z0-9\s.,]', '', texto)
    return texto.lower().strip()

def agrupar_por_linha(resultados, tolerancia_y=30):
    """
    Agrupa palavras que estão próximas verticalmente em linhas.
    """
    linhas = []
    for bbox, texto, conf in resultados:
        if conf < 0.2:
            continue
        x_min = min([p[0] for p in bbox])
        y_min = min([p[1] for p in bbox])
        colocado = False
        for linha in linhas:
            if abs(linha['y'] - y_min) <= tolerancia_y:
                linha['textos'].append((texto, x_min))
                linha['y'] = min(linha['y'], y_min)
                colocado = True
                break
        if not colocado:
            linhas.append({
                'y': y_min,
                'textos': [(texto, x_min)]
            })
    linhas = sorted(linhas, key=lambda l: l['y'])
    return linhas

def reconstruir_texto(linha):
    """
    Ordena as palavras pela posição X e retorna o texto concatenado.
    """
    palavras_ordenadas = sorted(linha['textos'], key=lambda t: t[1])
    return " ".join([t[0] for t in palavras_ordenadas])

def corrigir_preco(preco_texto):

    preco_texto = preco_texto.replace(',', '.').strip()
    digits = re.sub(r'\D', '', preco_texto)

    if len(digits) != 4:
        return None

    if '.' not in preco_texto:
        preco_texto = digits[:-3] + '.' + digits[-3:]

    return preco_texto

def extrair_precos_automatico(imagem):
    altura, largura = imagem.shape[:2]
    imagem = cv2.resize(imagem, (largura * 2, altura * 2), interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = cv2.dilate(bin_img, np.ones((2, 2), np.uint8), iterations=1)

    resultados = reader.readtext(bin_img, detail=1)

    print("\n===== TEXTOS RECONHECIDOS =====")
    for bbox, texto, conf in resultados:
        print(f"Texto reconhecido: '{texto}' | Confiança: {conf:.2f}")

    linhas = agrupar_por_linha(resultados)
    precos_list = []
    preco_regex = r'(\d+[.,]?\d{1,3})'

    for linha in linhas:
        linha_texto = normalizar_texto(reconstruir_texto(linha))
        precos_encontrados = re.findall(preco_regex, linha_texto)

        precos_encontrados = [corrigir_preco(p) for p in precos_encontrados if corrigir_preco(p) is not None]

        if not precos_encontrados:
            continue

        combustiveis_encontrados = []
        palavras = linha_texto.split()
        for palavra in palavras:
            for comb in COMBUSTIVEIS:
                if fuzz.partial_ratio(comb, palavra) > 60:
                    combustiveis_encontrados.append(comb.capitalize())

        if not combustiveis_encontrados:
            for p in precos_encontrados:
                precos_list.append({
                    "combustivel": f"Item_{len(precos_list) + 1}",
                    "preco": p
                })
        else:
            for p in precos_encontrados:
                precos_list.append({
                    "combustivel": combustiveis_encontrados[0],
                    "preco": p
                })

    return precos_list

@app.route("/upload", methods=["POST"])
def upload_imagem():
    if 'imagem' not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400

    file = request.files['imagem']
    npimg = np.frombuffer(file.read(), np.uint8)
    imagem = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if imagem is None:
        return jsonify({"erro": "Não foi possível ler a imagem"}), 400

    precos = extrair_precos_automatico(imagem)
    return jsonify(precos)

if __name__ == "__main__":
    app.run(debug=True)
