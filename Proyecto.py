import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Leer el dataset
data = pd.read_csv('planets.csv')
def fuerza_gravitacional(m1, m2, p1, p2):
    G = 6.67430e-11  # Constante gravitacional
    distancia = np.linalg.norm(p2 - p1)
    fuerza = G * m1 * m2 / distancia**2
    direccion = (p2 - p1) / distancia
    return fuerza * direccion
planetas = []

# Extraer información de los planetas
for index, row in data[:-1].iterrows():
    planetas.append({
        "nombre": row["Planet"],
        "color": row["Color"],
        "masa": row["Mass (10^24kg)"],
        "diametro": row["Diameter (km)"],
        "densidad": row["Density (kg/m^3)"],
        "gravedad": row["Surface Gravity(m/s^2)"],
        "velocidad_escape": row["Escape Velocity (km/s)"],
        "periodo_rotacion": row["Rotation Period (hours)"],
        "duracion_dia": row["Length of Day (hours)"],
        "distancia_sol": row["Distance from Sun (10^6 km)"],
        "perihelio": row["Perihelion (10^6 km)"],
        "afelio": row["Aphelion (10^6 km)"],
        "periodo_orbital": float(row["Orbital Period (days)"].replace(',', '')),
        "velocidad_orbital": row["Orbital Velocity (km/s)"],
        "inclinacion_orbital": row["Orbital Inclination (degrees)"],
        "excentricidad_orbital": row["Orbital Eccentricity"],
        "oblicuidad_orbita": row["Obliquity to Orbit (degrees)"],
        "temperatura_media": row["Mean Temperature (C)"],
        "presion_superficial": row["Surface Pressure (bars)"],
        "numero_lunas": row["Number of Moons"],
        "posicion": np.array([row["Distance from Sun (10^6 km)"] * 1e6, 0, 0]),
        "velocidad": np.array([0, row["Global Magnetic Field?"]])
    })

# Extraer información del asteroide (última fila del dataset)
asteroide_info = data.iloc[-1]
asteroide = {
    "nombre": asteroide_info["Planet"],

    "masa": asteroide_info["Mass (10^24kg)"],
    "diametro": asteroide_info["Diameter (km)"],
    "densidad": asteroide_info["Density (kg/m^3)"],
    "posicion": np.array([asteroide_info["Ring System?"], 0]),
    "velocidad": np.array([0, asteroide_info["Global Magnetic Field?"]])
}
import numpy as np

def calcular_posiciones(planetas, asteroide, num_iteraciones, dt):
    for planeta in planetas:
        planeta["posiciones"] = []
        velocidad_angular = 2 * np.pi / planeta["periodo_orbital"]
        for t in range(num_iteraciones):
            angulo = velocidad_angular * t * dt
            x = planeta["distancia_sol"] * np.cos(angulo)
            y = planeta["distancia_sol"] * np.sin(angulo)
            z = 0  # Puedes cambiar esto si quieres agregar una componente z
            planeta["posiciones"].append((x, y, z))

    asteroide["posiciones"] = []
    # Aquí necesitas calcular las posiciones del asteroide en función de su órbita y velocidad
    for t in range(num_iteraciones):
        x, y, z = 0, 0, 0  # Reemplaza esto con las posiciones calculadas del asteroide
        asteroide["posiciones"].append((x, y, z))

num_iteraciones = 1000  # Ajusta este número según tus necesidades
dt = 0.1  # Ajusta este valor según tus necesidades (representa el intervalo de tiempo entre los cálculos)

calcular_posiciones(planetas, asteroide, num_iteraciones, dt)

def actualizar_cuerpos(asteroide, planetas, dt):
    for planeta in planetas:
        fuerza = fuerza_gravitacional(asteroide["masa"], planeta["masa"], asteroide["posicion"], planeta["posicion"])
        asteroide["velocidad"] += (fuerza / asteroide["masa"]) * dt
        planeta["velocidad"] -= (fuerza / planeta["masa"]) * dt

    asteroide["posicion"] += asteroide["velocidad"] * dt
    for planeta in planetas:
        planeta["posicion"] += planeta["velocidad"] * dt

color_map = {
    'Brown and Grey': 'saddlebrown',
    'Blue, Brown Green and White': 'dodgerblue',
    'Red, Brown and Tan': 'darkred',
    'Brown, Orange and Tan, with White cloud stripes': 'peru',
    'Golden, Brown, and Blue-Grey': 'goldenrod',
    'Blue-Green': 'mediumaquamarine',
    # Agrega más mapeos si es necesario
}

from mpl_toolkits.mplot3d import Axes3D

def visualizar_simulacion(asteroide, planetas, num_iteraciones, dt):
    max_distancia = max([planeta["distancia_sol"] for planeta in planetas]) * 1.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-max_distancia, max_distancia)
    ax.set_ylim(-max_distancia, max_distancia)
    ax.set_zlim(-max_distancia, max_distancia)

    puntos = []
    for i, planeta in enumerate(planetas):
        color = color_map.get(planeta["color"], planeta["color"])
        tamaño = planeta["diametro"] / max([p["diametro"] for p in planetas]) * 100
        puntos.append(ax.plot([], [], [], marker="o", markersize=tamaño, color=color, label=planeta["nombre"])[0])
    puntos.append(ax.plot([], [], [], marker="o", color="gray", label=asteroide["nombre"])[0])

    def init():
        for punto in puntos:
            punto.set_data([], [])
            punto.set_3d_properties([])
        return puntos

    def update(frame):
        frame = frame % num_iteraciones
        lineas = [ax.plot([], [], [], c='gray')[0] for _ in range(len(planetas))]

        for i, (punto, linea) in enumerate(zip(puntos, lineas)):
            print(f"Planeta {i}, frame: {frame}")  # Agrega esta línea
            print(f"Longitud de posiciones: {len(planetas[i]['posiciones'])}")  # Agrega esta línea
            x, y, z = planetas[i]["posiciones"][frame]
            punto.set_data(x, y)
            punto.set_3d_properties(z)  # Añade la posición en el eje Z
        return puntos


    ax.legend(loc='upper left')
    ani = FuncAnimation(fig, update, frames=range(num_iteraciones), init_func=init, blit=True)
    plt.show()


visualizar_simulacion(asteroide, planetas, num_iteraciones, dt)



