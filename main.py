import numpy as np
import matplotlib.pyplot as plt

# Ustawienie stałego seeda dla powtarzalności wyników
np.random.seed(42)

# Stała lokalizacja przeszkody (w lewym dolnym rogu)
obstacle_x = [10, 80]  # x współrzędne
obstacle_y = [60, 60]  # y współrzędne (stałe, aby przeszkoda była linią poziomą)

# Liczba segmentów do narysowania
num_segments = 5  # Możemy mieć maksymalnie 5 odcinków

# Generowanie punktów w taki sposób, by unikały przeszkody
# Ograniczamy generowanie punktów do obszaru powyżej przeszkody
points = []
for i in range(num_segments):
    x1, y1 = np.random.uniform(90, 160), np.random.uniform(100, 250)  # Punkt początkowy
    x2, y2 = x1 + np.random.uniform(-30, 30), y1 + np.random.uniform(-30, 30)  # Punkt końcowy
    points.append(((x1, y1), (x2, y2)))

# Tworzenie wykresu
plt.figure(figsize=(10, 6))

# Rysowanie odcinków trasy
for i, (p1, p2) in enumerate(points):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', marker='o', markersize=10)
    plt.text(p1[0], p1[1], f'{i+1}a', fontsize=12, color='red', ha='right')
    plt.text(p2[0], p2[1], f'{i+1}b', fontsize=12, color='red', ha='right')

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
