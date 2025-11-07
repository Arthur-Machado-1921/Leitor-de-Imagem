import cv2
import numpy as np
import os

# Carrega imagem
imagem = cv2.imread("shell.jpg")
altura, largura = imagem.shape[:2]
imagem = cv2.resize(imagem, (largura*2, altura*2), interpolation=cv2.INTER_CUBIC)

# Pasta para salvar resultados
os.makedirs("resultados2", exist_ok=True)

# Converte para cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
kernel = np.ones((2,2), np.uint8)
contador = 1

def salvar(img, nome):
    global contador
    caminho = f"resultados2/{contador:02d}_{nome}.jpg"
    cv2.imwrite(caminho, img)
    contador += 1

# 1. Cinza simples
salvar(cinza, "cinza")

# 2. Equalização de histograma
eq = cv2.equalizeHist(cinza)
salvar(eq, "cinza_eq")

# 3. Gaussian Blur
blur_gauss = cv2.GaussianBlur(cinza, (3,3), 0)
salvar(blur_gauss, "gauss_blur")

# 4. Median Blur
blur_med = cv2.medianBlur(cinza, 3)
salvar(blur_med, "median_blur")

# 5. Binarização adaptativa
bin_adapt = cv2.adaptiveThreshold(cinza, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 4)
salvar(bin_adapt, "bin_adapt")

# 6. Otsu
_, otsu = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
salvar(otsu, "otsu_inv")
_, otsu_trunc = cv2.threshold(cinza, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
salvar(otsu_trunc, "otsu_trunc")
_, otsu_tozero = cv2.threshold(cinza, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
salvar(otsu_tozero, "otsu_tozero")

# 7. Dilatação / Erosão / abertura / fechamento
dil = cv2.dilate(bin_adapt, kernel, iterations=1)
ero = cv2.erode(bin_adapt, kernel, iterations=1)
open_img = cv2.morphologyEx(bin_adapt, cv2.MORPH_OPEN, kernel)
close_img = cv2.morphologyEx(bin_adapt, cv2.MORPH_CLOSE, kernel)
salvar(dil, "dilatacao")
salvar(ero, "erosao")
salvar(open_img, "abertura")
salvar(close_img, "fechamento")

# 8. Top-hat / Black-hat
tophat = cv2.morphologyEx(cinza, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(cinza, cv2.MORPH_BLACKHAT, kernel)
salvar(tophat, "tophat")
salvar(blackhat, "blackhat")

# 9. Sobel X e Y
sobelx = cv2.Sobel(cinza, cv2.CV_8U, 1, 0, ksize=3)
sobely = cv2.Sobel(cinza, cv2.CV_8U, 0, 1, ksize=3)
salvar(sobelx, "sobel_x")
salvar(sobely, "sobel_y")

# 10. Laplaciano
lap = cv2.Laplacian(cinza, cv2.CV_8U, ksize=3)
salvar(lap, "laplaciano")

# 11. Inversão de cores
invert = cv2.bitwise_not(cinza)
salvar(invert, "inversao")

# 12. Normalização
norm = cv2.normalize(cinza, None, 0, 255, cv2.NORM_MINMAX)
salvar(norm, "normalizacao")

# 13 a 35: Combinações morfológicas e blur
combos = [
    (bin_adapt, "bin"),
    (otsu, "otsu"),
    (otsu_trunc, "otsu_trunc"),
    (otsu_tozero, "otsu_tozero"),
    (blur_gauss, "gauss"),
    (blur_med, "median"),
    (eq, "eq")
]

for img, nome in combos:
    salvar(cv2.dilate(img, kernel, iterations=1), f"{nome}_dil")
    salvar(cv2.erode(img, kernel, iterations=1), f"{nome}_ero")
    salvar(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel), f"{nome}_open")
    salvar(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), f"{nome}_close")

print("Mega multi-técnica concluída! Veja a pasta 'resultados'")
