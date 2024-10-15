import numpy as np
import matplotlib.pyplot as plt

# Ustawienie stałego seeda dla powtarzalności wyników
np.random.seed(42)

# Nowa lokalizacja przeszkody (bliżej odcinków 2 i 3)
obstacle_x = [100, 140]  # x współrzędne
obstacle_y = [120, 120]  # y współrzędne (stałe, aby przeszkoda była linią poziomą)

# Liczba segmentów do narysowania
num_segments = 5  # Możemy mieć maksymalnie 5 odcinków

# Generowanie punktów w taki sposób, by unikały przeszkody
# Ograniczamy generowanie punktów do obszaru powyżej przeszkody
points = []
for i in range(num_segments):
    x1, y1 = np.random.uniform(90, 160), np.random.uniform(150, 250)  # Punkt początkowy
    x2, y2 = x1 + np.random.uniform(-30, 30), y1 + np.random.uniform(-30, 30)  # Punkt końcowy
    points.append(((x1, y1), (x2, y2)))

# Tworzenie wykresu
plt.figure(figsize=(10, 6))

# Rysowanie odcinków trasy (ciągłe linie)
for i, (p1, p2) in enumerate(points):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', marker='o', markersize=10)  # Zwykłe, ciągłe linie
    plt.text(p1[0], p1[1], f'{i+1}a', fontsize=12, color='red', ha='right')
    plt.text(p2[0], p2[1], f'{i+1}b', fontsize=12, color='red', ha='right')

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
plt.title('Trasa TSP z przeszkodą')

# Ustawienie zakresu osi
plt.xlim(0, 200)
plt.ylim(0, 300)

# Dodanie legendy
plt.legend()

# Dodanie siatki
plt.grid(True)

# Wyświetlenie wykresu
plt.show()
