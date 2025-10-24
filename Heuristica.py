# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:24:09 2025

@author: Francisco FV C
"""
# --------------------------------------------------------------------------------
# Importaciones
# --------------------------------------------------------------------------------
import math
import heapq

# --------------------------------------------------------------------------------
# Insertar datos del grafo
# --------------------------------------------------------------------------------

grafo = {("nodo_origen", "nodo_destino"): "peso"}

# --------------------------------------------------------------------------------
# 1) Función para calcular el costo de una ruta
# --------------------------------------------------------------------------------
def costo_ruta(tour):
    total = 0
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i+1]
        total += grafo[(u, v)]
    return total

# --------------------------------------------------------------------------------
# 2) Función para detectar subciclos en una ruta
# --------------------------------------------------------------------------------
def ciclos(ruta):
    lista_ciclos = []
    n = len(ruta)
    for inicio in range(n - 1):
        for fin in range(inicio + 1, n):
            if ruta[inicio] == ruta[fin]:
                sub_ciclo = ruta[inicio:fin + 1]
                # Requisitos: >= 3 nodos y no sea el tour entero
                if len(sub_ciclo) > 2 and not (inicio == 0 and fin == n - 1):
                    lista_ciclos.append(sub_ciclo)
    return lista_ciclos

# --------------------------------------------------------------------------------
# 3) Función para quitar un subciclo de una ruta
# --------------------------------------------------------------------------------
def quitar_sub_ciclo(ruta, ciclo):
    copia_ruta = ruta[:]
    str_ruta = ",".join(map(str, copia_ruta))
    str_ciclo = ",".join(map(str, ciclo))

    indice = str_ruta.find(str_ciclo)
    if indice == -1:
        return copia_ruta

    # Reemplazamos la primera ocurrencia
    ruta_sin_ciclo = str_ruta.replace(str_ciclo, "", 1)
    nodos = [p for p in ruta_sin_ciclo.split(",") if p != ""]
    try:
        ruta_final = list(map(int, nodos))
    except ValueError:
        return copia_ruta

    # Insertar último nodo del subciclo en la posición adecuada
    ultimo_nodo = ciclo[-1]
    partes_antes_de_ciclo = [p for p in str_ruta[:indice].split(",") if p]
    lugar_insertar = len(partes_antes_de_ciclo)
    if lugar_insertar > len(ruta_final):
        lugar_insertar = len(ruta_final)

    ruta_final.insert(lugar_insertar, ultimo_nodo)
    return ruta_final

# --------------------------------------------------------------------------------
# 4) Dijkstra para distancias mínimas
# --------------------------------------------------------------------------------
def dijkstra(grafo, inicio, fin):
    """
    Calcula la distancia mínima desde 'inicio' hasta 'fin'.
    Devuelve (distancia, ruta) usando un heap de prioridad.
    """
    nodos = set()
    for (u, v) in grafo:
        nodos.add(u)
        nodos.add(v)
    
    distancia = {n: float('inf') for n in nodos}
    predecesor = {n: None for n in nodos}
    distancia[inicio] = 0
    pq = []
    heapq.heappush(pq, (0, inicio))

    while pq:
        dist_actual, nodo_actual = heapq.heappop(pq)
        if nodo_actual == fin:
            break

        for (u, v), peso in grafo.items():
            if u == nodo_actual:
                nueva_dist = dist_actual + peso
                if nueva_dist < distancia[v]:
                    distancia[v] = nueva_dist
                    predecesor[v] = u
                    heapq.heappush(pq, (nueva_dist, v))

    # Reconstruir la ruta
    ruta = []
    nodo = fin
    while nodo is not None:
        ruta.append(nodo)
        nodo = predecesor[nodo]
    ruta.reverse()

    return distancia[fin], ruta

# --------------------------------------------------------------------------------
# 5) Añadir un subciclo a una ruta
# --------------------------------------------------------------------------------
def anadir_subciclo(ruta, subciclo, grafo):

    if not subciclo:
        return ruta[:]

    X = subciclo[0]

    # Buscar el nodo más cercano en la ruta a X
    mejor_nodo_entrada = None
    mejor_dist_entrada = float('inf')
    mejor_ruta_entrada = None

    for nodo in ruta:
        dist, ruta_entrada = dijkstra(grafo, nodo, X)
        if dist < mejor_dist_entrada:
            mejor_dist_entrada = dist
            mejor_nodo_entrada = nodo
            mejor_ruta_entrada = ruta_entrada

    if mejor_nodo_entrada is None or mejor_ruta_entrada is None:
        return ruta

    idx = ruta.index(mejor_nodo_entrada)
    nodos_a_insertar = subciclo[:]
    ultima_parada = nodos_a_insertar[-1]

    # Camino de regreso usando Dijkstra
    mejor_nodo_salida = mejor_nodo_entrada
    _, camino_de_vuelta = dijkstra(grafo, ultima_parada, mejor_nodo_salida)
    if camino_de_vuelta is None:
        return ruta

    # Evitar duplicaciones
    if mejor_ruta_entrada[-1] == X:
        mejor_ruta_entrada = mejor_ruta_entrada[:-1]
    if camino_de_vuelta[0] == ultima_parada:
        camino_de_vuelta = camino_de_vuelta[1:]

    nueva_ruta = (
        ruta[:idx]
        + mejor_ruta_entrada
        + nodos_a_insertar
        + camino_de_vuelta
        + ruta[idx+1:]
    )
    return nueva_ruta

# --------------------------------------------------------------------------------
# 6) Generar vecinos con cycle_trade
# --------------------------------------------------------------------------------
def cycle_trade(T):
    vecinos = []

    # Localizar la ruta más "cara" y la más "barata"
    idx_caro = max(range(len(T)), key=lambda i: costo_ruta(T[i]))
    idx_barato = min(range(len(T)), key=lambda i: costo_ruta(T[i]))

    t1 = T[idx_caro]
    t2 = T[idx_barato]
    subciclos = ciclos(t1)

    for c in subciclos:
        t1_nuevo = quitar_sub_ciclo(t1, c)
        t2_nuevo = anadir_subciclo(t2, c, grafo)

        # Construimos la nueva solución
        T_vecino = []
        for i, tour in enumerate(T):
            if i != idx_caro and i != idx_barato:
                T_vecino.append(tour)
        T_vecino.append(t1_nuevo)
        T_vecino.append(t2_nuevo)

        # Almacenamos la transición
        transicion = (tuple(c), idx_caro, idx_barato)
        vecinos.append((T_vecino, transicion))

    return vecinos

# --------------------------------------------------------------------------------
# 7) Función objetivo para Weighted k-CPP
# --------------------------------------------------------------------------------
def funcion_objetivo(solucion, lambda_):
    """
    FO = lambda * max(costo_ruta(t)) + (1 - lambda) * sum(costo_ruta(t))
    """
    costos = [costo_ruta(t) for t in solucion]
    return lambda_ * max(costos) + (1 - lambda_) * sum(costos)

# --------------------------------------------------------------------------------
# 8) Búsqueda Tabú con cycle_trade
# --------------------------------------------------------------------------------
def tabu_search_cycle_trade(solucion_inicial, lambda_, 
                            max_iter, max_sin_mejora, 
                            max_tabu_size):
    mejor_sol = solucion_inicial
    mejor_fo = funcion_objetivo(mejor_sol, lambda_)

    sol_actual = solucion_inicial
    fo_actual = mejor_fo

    tabu_list = []
    sin_mejora = 0

    for it in range(max_iter):
        print(f"\n[Iter {it}] Sol. actual = {sol_actual}")
        print(f"[Iter {it}] FO actual = {fo_actual}, FO mejor = {mejor_fo}")

        vecinos = cycle_trade(sol_actual)
        if not vecinos:
            print(f"[Iter {it}] No hay vecinos, se detiene.")
            break

        mejor_vecino = None
        mejor_vecino_fo = math.inf
        mejor_vecino_transicion = None

        for (sol_vecina, transicion) in vecinos:
            # Checar si la transición está en tabu_list
            if transicion in tabu_list:
                # Criterio de aspiración
                fo_vecina = funcion_objetivo(sol_vecina, lambda_)
                if fo_vecina < mejor_fo:
                    if fo_vecina < mejor_vecino_fo:
                        mejor_vecino = sol_vecina
                        mejor_vecino_fo = fo_vecina
                        mejor_vecino_transicion = transicion
                else:
                    continue
            else:
                fo_vecina = funcion_objetivo(sol_vecina, lambda_)
                if fo_vecina < mejor_vecino_fo:
                    mejor_vecino = sol_vecina
                    mejor_vecino_fo = fo_vecina
                    mejor_vecino_transicion = transicion

        if mejor_vecino is None:
            print(f"[Iter {it}] No hay vecinos válidos, se detiene.")
            break

        # Actualizamos solución actual
        sol_actual = mejor_vecino
        fo_actual = mejor_vecino_fo

        # Agregamos la transición a la lista tabú
        if mejor_vecino_transicion:
            tabu_list.append(mejor_vecino_transicion)
            if len(tabu_list) > max_tabu_size:
                tabu_list.pop(0)

        if fo_actual < mejor_fo:
            mejor_sol = sol_actual
            mejor_fo = fo_actual
            sin_mejora = 0
        else:
            sin_mejora += 1

        if sin_mejora >= max_sin_mejora:
            print(f"[Iter {it}] Alcanzadas {max_sin_mejora} iteraciones sin mejora. Fin.")
            break

    return mejor_sol, mejor_fo

# --------------------------------------------------------------------------------
# 9) Ejemplo de uso
# --------------------------------------------------------------------------------
T_inicial = [ ["ruta_i"],
             ["ruta_i+1"]
    ]


    
print("\n=== Iniciando Tabu Search con cycle_trade ===\n")
mejor_sol, mejor_fo = tabu_search_cycle_trade(
    T_inicial,
    lambda_=1,
    max_iter=5000,
    max_sin_mejora=500,
    max_tabu_size=20
)
    
print("\n=== RESULTADO FINAL ===")
for i, ruta in enumerate(mejor_sol, start=1):
    c = costo_ruta(ruta)
    print(f"  Ruta {i}: {ruta}  |  Costo = {c}")
print(f"Mejor Función Objetivo (FO) = {mejor_fo}")


def analisis_sensibilidad_lambda(T_inicial, valores_lambda, 
                                max_iter=500, max_sin_mejora=10, 
                                max_tabu_size=10):
    resultados = []

    for lam in valores_lambda:
        print(f"\n=== Analizando lambda = {lam:.2f} ===")
        mejor_sol, mejor_fo = tabu_search_cycle_trade(
            T_inicial,
            lambda_=lam,
            max_iter=max_iter,
            max_sin_mejora=max_sin_mejora,
            max_tabu_size=max_tabu_size
        )
        # calcular métricas reales
        costos = [costo_ruta(r) for r in mejor_sol]
        costo_max = max(costos)
        costo_total = sum(costos)

        resultados.append({
            "lambda": lam,
            "costo_max": costo_max,
            "costo_total": costo_total,
            "solucion": mejor_sol
        })

        print(f"  -> Ruta más cara = {costo_max}")
        print(f"  -> Costo total   = {costo_total}")

    return resultados


# Ejemplo de uso
valores_lambda = [i/10 for i in range(0, 11)]  # 0.0 a 1.0
resultados = analisis_sensibilidad_lambda(T_inicial, valores_lambda)

# (Opcional) Graficar sensibilidad
import matplotlib.pyplot as plt

lambdas = [r["lambda"] for r in resultados]
costo_max = [r["costo_max"] for r in resultados]
costo_total = [r["costo_total"] for r in resultados]

plt.figure(figsize=(8,5))
plt.plot(lambdas, costo_max, marker="o", label="Ruta más cara")
plt.plot(lambdas, costo_total, marker="s", label="Costo total")
plt.xlabel("λ")
plt.ylabel("Costo")
plt.title("Análisis de sensibilidad respecto a λ")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------------------------------------------
# 9) Análisis de sensibilidad de parámetros de Tabu Search
# --------------------------------------------------------------------------------
def analisis_parametros_tabu(T_inicial, valores_tabu, valores_iter, valores_sin_mejora):
    resultados = []
    for tabu_size in valores_tabu:
        for max_iter in valores_iter:
            for sin_mej in valores_sin_mejora:
                print(f"\n>>> Tabu={tabu_size}, Iter={max_iter}, SinMej={sin_mej}")
                mejor_sol, _ = tabu_search_cycle_trade(
                    T_inicial,
                    lambda_=1,  # o el λ que quieras analizar
                    max_iter=max_iter,
                    max_sin_mejora=sin_mej,
                    max_tabu_size=tabu_size
                )
                costos = [costo_ruta(r) for r in mejor_sol]
                resultados.append({
                    "tabu_size": tabu_size,
                    "max_iter": max_iter,
                    "sin_mej": sin_mej,
                    "costo_max": max(costos),
                    "costo_total": sum(costos)
                })
    return resultados


# --------------------------------------------------------------------------------
# 10) Ejemplo de uso del análisis de parámetros
# --------------------------------------------------------------------------------
valores_tabu = [5, 10, 20]
valores_iter = [500, 11000]
valores_sin_mejora = [5, 10]

resultados_param = analisis_parametros_tabu(T_inicial, valores_tabu, valores_iter, valores_sin_mejora)

import pandas as pd
df = pd.DataFrame(resultados_param)
print(df)

