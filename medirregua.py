import cv2
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'IMG_5301.mov')

if not os.path.exists(video_path):
    print(f"\n[ERRO CRÍTICO] O arquivo 'IMG_5301.mov' NÃO foi encontrado em:")
    print(f" -> {script_dir}")
    sys.exit()

cap = cv2.VideoCapture(video_path)
ret, frame_original = cap.read()
cap.release()

if not ret or frame_original is None:
    print("\n[ERRO] Não foi possível ler o vídeo.")
    sys.exit()

LARGURA_TELA = 800
altura_orig, largura_orig = frame_original.shape[:2]
fator_escala_visual = LARGURA_TELA / float(largura_orig)
altura_tela = int(altura_orig * fator_escala_visual)

frame_redimensionado = cv2.resize(frame_original, (LARGURA_TELA, altura_tela))
frame_draw = frame_redimensionado.copy()

pontos = []

def clicar(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pontos.append((x, y))
        print(f"Ponto clicado na tela: x={x}, y={y}")
        cv2.circle(frame_draw, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Clique nas duas pontas da regua (Ajustado a tela)", frame_draw)
        
        if len(pontos) == 2:
            p1, p2 = pontos[0], pontos[1]
            
            dist_px_redim = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            dist_px_real = dist_px_redim / fator_escala_visual
            
            print(f"LARGURA DA RÉGUA EM PIXELS: {dist_px_real:.2f} px")

cv2.imshow("Clique nas duas pontas da regua (Ajustado a tela)", frame_draw)
cv2.setMouseCallback("Clique nas duas pontas da regua (Ajustado a tela)", clicar)

print("\n[INSTRUÇÕES] A janela foi ajustada para caber inteira na sua tela.")
print("Clique no início da régua e depois no final dela.")
print("Pressione qualquer tecla no teclado para fechar a janela quando terminar.")

cv2.waitKey(0)
cv2.destroyAllWindows()
