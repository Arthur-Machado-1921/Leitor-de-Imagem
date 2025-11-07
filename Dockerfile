# Imagem base com Python
FROM python:3.10-slim

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema necessárias pro OpenCV e EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório da aplicação
WORKDIR /app

# Copia os arquivos de dependência e instala
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

# Expõe a porta do Flask
EXPOSE 5000

# Comando para rodar o Flask
CMD ["python", "leitor.py"]
