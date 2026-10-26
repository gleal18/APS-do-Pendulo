import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, 'Pendulo.mov')

L = 0.60                                       
g = 9.81                      
distancia_regua_m = 0.30      

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Não foi possível abrir o vídeo no caminho: {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0

lower_color = np.array([15, 60, 60])
upper_color = np.array([55, 255, 255])

times, x_pixels, y_pixels = [], [], []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 8:  
            t = frame_count / fps
            
            times.append(t)
            x_pixels.append(x)
            y_pixels.append(y)

    frame_count += 1

cap.release()

if len(x_pixels) == 0:
    print("\nNenhum objeto foi detectado com a faixa de cores atual.")
    sys.exit()

t_data = np.array(times)
t_data = t_data - t_data[0]  

x_px = np.array(x_pixels)
y_px = np.array(y_pixels)

pixels_regua_estatica = 637.21 

distancia_regua_m = 0.30       

escala_m_px = distancia_regua_m / pixels_regua_estatica

x_m = (x_px - np.mean(x_px)) * escala_m_px
y_m = (y_px - np.mean(y_px)) * escala_m_px

df = pd.DataFrame({
    't (s)': np.round(t_data, 3),
    'x (m)': np.round(x_m, 4),
    'y (m)': np.round(y_m, 4)
})

output_csv = os.path.join(script_dir, 'tabela_pendulo_laranja.csv')

df.to_csv(output_csv, index=False, sep=';', decimal=',')
print(f"\nTabela compatível com Excel salva em: {output_csv}")

def oscilador_amortecido(t, A, gamma, omega, phi):
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)

p0 = [np.max(x_m), 0.02, 2 * np.pi / 1.5, 0]

try:
    popt, _ = curve_fit(oscilador_amortecido, t_data, x_m, p0=p0)
    A_fit, gamma_fit, omega_fit, phi_fit = popt

    omega_0_exp = np.sqrt(omega_fit**2 + gamma_fit**2)
    omega_0_teo = np.sqrt(g / L)
    Q = omega_0_exp / (2 * gamma_fit)

    print(f"Amplitude Inicial (A): {A_fit:.4f} m")
    print(f"Amortecimento (gamma): {gamma_fit:.4f} s^-1")
    print(f"Frequência Angular Ajustada (w_exp): {omega_fit:.4f} rad/s")
    print(f"Frequência Angular Teórica (w0_teo): {omega_0_teo:.4f} rad/s")
    print(f"Fase Inicial (phi): {phi_fit:.4f} rad")
    print(f"FATOR DE QUALIDADE (Q): {Q:.2f}")

    plt.figure(figsize=(10, 5))
    plt.scatter(t_data, x_m, s=8, color='orange', label='Dados Medidos (OpenCV)')
    plt.plot(t_data, oscilador_amortecido(t_data, *popt), 'b-', label='Ajuste Teórico (Curve Fit)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Posição Horizontal x (m)')
    plt.title(f'Oscilação do Pêndulo — Fator Q = {Q:.2f}')
    plt.legend()
    plt.grid(True)
    
    graph_path = os.path.join(script_dir, 'grafico_oscilador.png')
    plt.savefig(graph_path)
    plt.show()

except Exception as e:
    print(f"Erro no ajuste da curva: {e}")