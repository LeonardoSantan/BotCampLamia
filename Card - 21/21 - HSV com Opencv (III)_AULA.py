import cv2
import numpy as np

# Inicializa a captura de vídeo utilizando a câmera padrão (índice 0).
cap = cv2.VideoCapture(0)


# Função para aplicar uma máscara de cor ao frame HSV
# Recebe o frame HSV e os limites inferior e superior da cor.
def apply_mask(hsv_frame, low_color, high_color):
    # Cria uma máscara binária onde os pixels dentro do intervalo ficam brancos (255) e os demais pretos (0).
    mask = cv2.inRange(hsv_frame, low_color, high_color)
    # Aplica a máscara ao frame original usando uma operação de "bitwise_and".
    return cv2.bitwise_and(frame, frame, mask=mask)


# Loop principal para captura de vídeo em tempo real
while True:
    # Captura o frame atual da câmera.
    ret, frame = cap.read()

    # Verifica se o frame foi capturado corretamente. Se não, encerra o loop.
    if not ret:
        print("Erro ao capturar o vídeo")
        break

    # Converte o frame de BGR (padrão do OpenCV) para o espaço de cores HSV.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os intervalos de cores HSV para diferentes cores.
    vermelho_baixo = np.array([161, 155, 84])
    vermelho_alto = np.array([179, 255, 255])

    azul_baixo = np.array([94, 80, 2])
    azul_alto = np.array([126, 255, 255])

    verde_baixo = np.array([25, 52, 72])
    verde_alto = np.array([102, 255, 255])

    amarelo_baixo = np.array([18, 94, 140])
    amarelo_alto = np.array([35, 255, 255])

    laranja_baixo = np.array([10, 100, 20])
    laranja_alto = np.array([25, 255, 255])

    roxo_baixo = np.array([129, 50, 70])
    roxo_alto = np.array([158, 255, 255])

    rosa_baixo = np.array([160, 100, 100])
    rosa_alto = np.array([179, 255, 255])

    ciano_baixo = np.array([85, 50, 50])
    ciano_alto = np.array([100, 255, 255])

    marrom_baixo = np.array([10, 100, 20])
    marrom_alto = np.array([20, 255, 200])

    menos_branco_baixo = np.array([0, 42, 0])
    menos_branco_alto = np.array([179, 255, 255])

    # Aplica as máscaras para cada cor definida anteriormente.
    vermelho = apply_mask(hsv_frame, vermelho_baixo, vermelho_alto)
    azul = apply_mask(hsv_frame, azul_baixo, azul_alto)
    verde = apply_mask(hsv_frame, verde_baixo, verde_alto)
    amarelo = apply_mask(hsv_frame, amarelo_baixo, amarelo_alto)
    laranja = apply_mask(hsv_frame, laranja_baixo, laranja_alto)
    roxo = apply_mask(hsv_frame, roxo_baixo, roxo_alto)
    rosa = apply_mask(hsv_frame, rosa_baixo, rosa_alto)
    ciano = apply_mask(hsv_frame, ciano_baixo, ciano_alto)
    marrom = apply_mask(hsv_frame, marrom_baixo, marrom_alto)
    resultado = apply_mask(hsv_frame, menos_branco_baixo, menos_branco_alto)

    # Exibe o frame original e as máscaras aplicadas em janelas separadas.
    cv2.imshow("Frame Original", frame)
    cv2.imshow("Vermelho", vermelho)
    cv2.imshow("Azul", azul)
    cv2.imshow("Verde", verde)
    cv2.imshow("Amarelo", amarelo)
    cv2.imshow("Laranja", laranja)
    cv2.imshow("Roxo", roxo)
    cv2.imshow("Rosa", rosa)
    cv2.imshow("Ciano", ciano)
    cv2.imshow("Marrom", marrom)
    cv2.imshow("Tudo menos branco", resultado)

    # Aguarda pressionamento de tecla; encerra se a tecla ESC (código 27) for pressionada.
    key = cv2.waitKey(1)
    if key == 27:
        break

# Libera os recursos da câmera e fecha todas as janelas.
cap.release()
cv2.destroyAllWindows()
