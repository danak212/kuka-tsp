import numpy as np
import matplotlib.pyplot as plt

# Ustawienie stałego seeda dla powtarzalności wyników
np.random.seed(42)

# Lokalizacja przeszkody (nad punktem 2b, pod punktem 3b)
obstacle_x = [0, 105]  # x współrzędne (przeszkoda idzie aż do lewej ściany)
obstacle_y = [300, 215]  # y współrzędne, aby znajdowała się między 2b a 3b

# Liczba segmentów do narysowania
num_segments = 5  # Możemy mieć maksymalnie 5 odcinków

# Generowanie punktów w taki sposób, by unikały przeszkody
points = []
for i in range(num_segments):
    x1, y1 = np.random.uniform(90, 160), np.random.uniform(160, 250)  # Punkt początkowy
    x2, y2 = x1 + np.random.uniform(-40, 40), y1 + np.random.uniform(-55, 55)  # Punkt końcowy, większy rozrzut
    points.append(((x1, y1), (x2, y2)))

# Tworzenie wykresu
plt.figure(figsize=(10, 6))

# Rysowanie odcinków trasy (ciągłe linie)
for i, (p1, p2) in enumerate(points):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', marker='o', markersize=10)  # Zwykłe, ciągłe linie
    plt.text(p1[0], p1[1], f'{i+1}a', fontsize=12, color='red', ha='right')
    plt.text(p2[0], p2[1], f'{i+1}b', fontsize=12, color='red', ha='right')

    # MOŻLIWOŚĆ DODANIA WSPÓŁRZĘDNYCH PUNKTÓW
    # plt.text(p1[0], p1[1], f'{i+1}a ({p1[0]:.1f}, {p1[1]:.1f})', fontsize=12, color='red', ha='right')
    # plt.text(p2[0], p2[1], f'{i+1}b ({p2[0]:.1f}, {p2[1]:.1f})', fontsize=12, color='red', ha='right')

# Rysowanie połączeń między odcinkami (przerywane linie)
for i in range(len(points) - 1):
    p2_prev = points[i][1]  # Koniec poprzedniego odcinka
    p1_next = points[i + 1][0]  # Początek następnego odcinka
    plt.plot([p2_prev[0], p1_next[0]], [p2_prev[1], p1_next[1]], 'g--')  # Połączenie przerywane

# Rysowanie przeszkody jako czerwonej linii
plt.plot(obstacle_x, obstacle_y, 'r-', linewidth=4, label='Przeszkoda')

# Oznaczenia osi
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('Problem rysowania odcinków z przeszkodą')

# Ustawienie zakresu osi
plt.xlim(0, 200)
plt.ylim(0, 300)

# Dodanie legendy
plt.legend()

# Dodanie siatki
plt.grid(True)

# Wyświetlenie wykresu
plt.show()