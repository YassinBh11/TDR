import random
import time
import os
import pyfiglet
import torch 
import torch.nn as nn
import numpy as np

class Red_neuronal(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, output_size=7):
        super(Red_neuronal, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ===== FUNCIONES PARA LA IA =====
def acciones_validas_ia(posiciones, meta_index, jugador):
    """Devuelve lista de índices de acciones válidas para el jugador especificado"""
    action_to_move = {
        0: "m1", 1: "m2", 2: "m3",
        3: "p1", 4: "p2", 5: "p3", 6: "p4"
    }

    valid = []
    for act_idx, mov in action_to_move.items():
        if mov.startswith("m"):
            pieza = f"{jugador}_{mov}"
            if posiciones[pieza] < meta_index:
                valid.append(act_idx)
        else:
            pieza = mov
            pos_ping = posiciones.get(pieza)
            if pos_ping is None or pos_ping >= meta_index:
                continue
            
            puede_mover = False
            for nm, p in posiciones.items():
                if nm.startswith(f"{jugador}_m") and p == pos_ping:
                    puede_mover = True
                    break
            
            if puede_mover:
                valid.append(act_idx)
    
    return valid if valid else [0, 1, 2]


def obtener_estado_juego(posiciones, tablero):
    """Convierte el estado del juego al formato que espera la IA"""
    return np.array([
        posiciones["j1_m1"],
        posiciones["j1_m2"],
        posiciones["j1_m3"],
        posiciones["j2_m1"],
        posiciones["j2_m2"],
        posiciones["j2_m3"],
        posiciones["p1"],
        posiciones["p2"],
        posiciones["p3"],
        posiciones["p4"],
        len(tablero),
    ], dtype=np.float32)


def seleccionar_accion_ia(estado, modelo, posiciones, meta_index, jugador):
    """Selecciona la mejor acción válida usando el modelo entrenado"""
    acciones_validas = acciones_validas_ia(posiciones, meta_index, jugador)
    
    if not acciones_validas:
        return 0
    
    with torch.no_grad():
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        q_values = modelo(estado_tensor).squeeze(0)
    
    q_validas = [(i, q_values[i].item()) for i in acciones_validas]
    mejor_accion = max(q_validas, key=lambda x: x[1])[0]
    
    return mejor_accion


def traducir_accion_ia(accion_idx):
    """Convierte índice de acción (0-6) a nombre de movimiento"""
    action_to_move = {
        0: "m1", 1: "m2", 2: "m3",
        3: "p1", 4: "p2", 5: "p3", 6: "p4"
    }
    return action_to_move.get(accion_idx, "m1")


def cargar_modelo_ia(ruta="modelo_dqn_estrategia_mental_10.pth"):
    """Carga el modelo entrenado"""
    try:
        checkpoint = torch.load(ruta)
        modelo = Red_neuronal(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size']
        )
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.eval()
        print(f"✅ Modelo cargado: {ruta}")
        return modelo
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró '{ruta}'")
        return None
    except Exception as e:
        print(f"❌ ERROR al cargar modelo: {e}")
        return None


def limpiar():
    os.system("cls" if os.name=="nt" else "clear")


def todos_en_meta(jugador, posiciones, meta_index):
    return all(posiciones[f"{jugador}_m{i}"] >= meta_index for i in range(1, 4))


def poder_mover_pingorote(nombre_pingorote, posiciones):
    posicion_pingorote = posiciones[nombre_pingorote]
    for nombre, pos in posiciones.items():
        if (("m" in nombre) and pos == posicion_pingorote):
            return True
    return False


def eleccion_muñecos_ia(posiciones, meta_index, modelo_ia, jugador, tablero):
    """Versión genérica de elección para cualquier jugador"""
    if todos_en_meta(jugador, posiciones, meta_index):
        return None
    
    estado = obtener_estado_juego(posiciones, tablero)
    accion_idx = seleccionar_accion_ia(estado, modelo_ia, posiciones, meta_index, jugador)
    movimiento = traducir_accion_ia(accion_idx)
    
    return movimiento


def movimientos(eleccion, dado, posiciones, jugador, meta_index, modelo_ia, tablero):
    while True:
        if eleccion in ["m1", "m2", "m3"]:
            posiciones[f"{jugador}_{eleccion}"] += dado
            break
        elif eleccion in ["p1", "p2", "p3", "p4"]:
            if poder_mover_pingorote(eleccion, posiciones):
                posiciones[eleccion] += dado
                break
            else:
                eleccion = eleccion_muñecos_ia(posiciones, meta_index, modelo_ia, jugador, tablero)
                if eleccion is None:
                    break
        else:
            eleccion = eleccion_muñecos_ia(posiciones, meta_index, modelo_ia, jugador, tablero)
            if eleccion is None:
                break
    return posiciones


def mostrar_tablero(tablero, posiciones, verbose=True):
    if not verbose:
        return
    print("\n Estado del tablero:")
    for i, casilla in enumerate(tablero):
        if casilla == "Vacia":
            continue
        contenido = f"Casilla {i}: {casilla}"
        muñecos_en_casilla = [nombre for nombre, pos in posiciones.items() if pos == i]
        if muñecos_en_casilla:
            contenido += " ← " + ", ".join(muñecos_en_casilla)
        print(contenido)


def limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_ocupadas_antes):
    posiciones_actuales = set(posiciones.values())
    nuevas_casillas = []
    mapa_indices = {}
    nuevo_indice = 0

    for i, casilla in enumerate(tablero):
        if (i in casillas_ocupadas_antes and i not in posiciones_actuales and casilla != "META"):
            continue
        nuevas_casillas.append(casilla)
        mapa_indices[i] = nuevo_indice
        nuevo_indice += 1

    for nombre in posiciones:
        posiciones[nombre] = mapa_indices[posiciones[nombre]]

    nuevas_ocupadas = {mapa_indices[i] for i in casillas_ocupadas_antes if i in mapa_indices}
    nuevo_ultimo = {mapa_indices[i]: ocupante for i, ocupante in ultimo_ocupante.items() if i in mapa_indices}
    meta_index = len(nuevas_casillas) - 1

    return nuevas_casillas, nuevas_ocupadas, nuevo_ultimo, meta_index


def registrar_llegada(nombre, orden_llegada):
    """Registra el orden en que los muñecos llegan a META"""
    if nombre not in orden_llegada:
        orden_llegada.append(nombre)


def verificar_meta(posiciones, meta_index, orden_llegada):
    for nombre, pos in posiciones.items():
        if nombre.startswith("j"):
            if pos >= meta_index:
                posiciones[nombre] = meta_index
                registrar_llegada(nombre, orden_llegada)


def marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante):
    pos_nueva = posiciones[pieza_movida]
    if pieza_movida.startswith("j1_"):
        propietario = "j1"
    elif pieza_movida.startswith("j2_"):
        propietario = "j2"
    else:
        propietario = None
    if propietario is not None:
        ultimo_ocupante[pos_nueva] = propietario


def calcular_puntuacion_final(inventario):
    inventario = inventario.copy()
    negativos = [v for v in inventario if isinstance(v, int) and v < 0]

    for i, valor in enumerate(inventario):
        if valor == "SUERTE" and negativos:
            mayor_negativo = min(negativos)
            idx_neg = inventario.index(mayor_negativo)
            inventario[idx_neg] = abs(mayor_negativo)
            negativos.remove(mayor_negativo)

    puntuacion = sum(v for v in inventario if isinstance(v, int))
    return puntuacion


def mostrar_inventarios(inventario_j1, inventario_j2, verbose=True):
    if not verbose:
        return
    print(f"\nInventario j1: {inventario_j1}")
    print(f"Inventario j2: {inventario_j2}")


# ============================================
# FUNCIÓN PRINCIPAL PARA JUGAR UNA PARTIDA
# ============================================
def jugar_partida(modelo_ia_1, modelo_ia_2, verbose=False):
    """Juega una partida completa y retorna el ganador"""
    
    # Inicializar tablero
    tablero = ["INICIO"]
    for i in range(-1, -5, -1):
        tablero.append(i)
    for _ in range(2):
        tablero.append("SUERTE")
    for i in range(5, 1, -1):
        tablero.append(i)
    for i in range(-1, -6, -1):
        tablero.append(i)
    tablero.append("META")
    
    # Inicializar posiciones
    posiciones = {
        "j1_m1": 0, "j1_m2": 0, "j1_m3": 0,
        "j2_m1": 0, "j2_m2": 0, "j2_m3": 0,
        "p1": 5, "p2": 6, "p3": 7, "p4": 8,
    }
    
    meta_index = len(tablero) - 1
    ultimo_ocupante = {}
    inventario_j1 = []
    inventario_j2 = []
    orden_llegada = []
    
    # Determinar quién empieza
    dadoj1 = random.randint(1, 6)
    dadoj2 = random.randint(1, 6)
    
    while dadoj1 == dadoj2:
        dadoj1 = random.randint(1, 6)
        dadoj2 = random.randint(1, 6)
    
    if dadoj1 < dadoj2:
        primero, segundo = "j1", "j2"
    else:
        primero, segundo = "j2", "j1"
    
    turno = 1
    
    # Bucle principal de la partida
    while any(nombre.startswith("j") and pos != meta_index for nombre, pos in posiciones.items()):
        
        # === TURNO DEL PRIMERO ===
        if primero == "j1":
            modelo_actual = modelo_ia_1
        else:
            modelo_actual = modelo_ia_2
        
        if not todos_en_meta(primero, posiciones, meta_index):
            dado = random.randint(1, 6)
            eleccion = eleccion_muñecos_ia(posiciones, meta_index, modelo_actual, primero, tablero)
            
            if eleccion is not None:
                pieza_movida = f"{primero}_{eleccion}" if eleccion.startswith("m") else eleccion
                posicion_origen = posiciones[pieza_movida]
                
                posiciones = movimientos(eleccion, dado, posiciones, primero, meta_index, modelo_actual, tablero)
                marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante)
                
                ocupantes_origen = [p for p, pos in posiciones.items() if pos == posicion_origen]
                casillas_vacias = set()
                
                if len(ocupantes_origen) == 0 and posicion_origen != meta_index:
                    casillas_vacias.add(posicion_origen)
                    valor_casilla = tablero[posicion_origen]
                    
                    if valor_casilla not in ("INICIO", "META"):
                        if primero == "j1":
                            inventario_j1.append(valor_casilla)
                        else:
                            inventario_j2.append(valor_casilla)
                
                for nombre in posiciones:
                    if nombre.startswith("j") or nombre.startswith("p"):
                        posiciones[nombre] = min(posiciones[nombre], meta_index)
                
                tablero, _, ultimo_ocupante, meta_index = limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_vacias)
                verificar_meta(posiciones, meta_index, orden_llegada)
        
        # === TURNO DEL SEGUNDO ===
        if segundo == "j1":
            modelo_actual = modelo_ia_1
        else:
            modelo_actual = modelo_ia_2
        
        if not todos_en_meta(segundo, posiciones, meta_index):
            dado = random.randint(1, 6)
            eleccion = eleccion_muñecos_ia(posiciones, meta_index, modelo_actual, segundo, tablero)
            
            if eleccion is not None:
                pieza_movida = f"{segundo}_{eleccion}" if eleccion.startswith("m") else eleccion
                posicion_origen = posiciones[pieza_movida]
                
                posiciones = movimientos(eleccion, dado, posiciones, segundo, meta_index, modelo_actual, tablero)
                marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante)
                
                ocupantes_origen = [p for p, pos in posiciones.items() if pos == posicion_origen]
                casillas_vacias = set()
                
                if len(ocupantes_origen) == 0 and posicion_origen != meta_index:
                    casillas_vacias.add(posicion_origen)
                    valor_casilla = tablero[posicion_origen]
                    
                    if valor_casilla not in ("INICIO", "META"):
                        if segundo == "j1":
                            inventario_j1.append(valor_casilla)
                        else:
                            inventario_j2.append(valor_casilla)
                
                for nombre in posiciones:
                    if nombre.startswith("j") or nombre.startswith("p"):
                        posiciones[nombre] = min(posiciones[nombre], meta_index)
                
                tablero, _, ultimo_ocupante, meta_index = limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_vacias)
                verificar_meta(posiciones, meta_index, orden_llegada)
        
        turno += 1
        if verbose and turno % 10 == 0:
            print(f"Turno {turno}...")
    
    # Calcular puntuaciones finales
    puntuacion1 = calcular_puntuacion_final(inventario_j1)
    puntuacion2 = calcular_puntuacion_final(inventario_j2)
    
    recompensas = [5, 4, 3, 2, 1, 0]
    for i, muñeco in enumerate(orden_llegada):
        puntos = recompensas[min(i, len(recompensas) - 1)]
        if muñeco.startswith("j1_"):
            puntuacion1 += puntos
        elif muñeco.startswith("j2_"):
            puntuacion2 += puntos
    
    if verbose:
        print(f"\nPuntuación j1: {puntuacion1}")
        print(f"Puntuación j2: {puntuacion2}")
    
    if puntuacion1 > puntuacion2:
        return "j1"
    elif puntuacion2 > puntuacion1:
        return "j2"
    else:
        return "empate"


# ============================================
# PROGRAMA PRINCIPAL
# ============================================

# Cargar modelos
modelo_ia_1 = cargar_modelo_ia("modelo_dqn_estrategia_mental_1000.pth")
modelo_ia_2 = cargar_modelo_ia("modelo_dqn_100k.pth")

if modelo_ia_1 is None or modelo_ia_2 is None:
    print("Error: No se pudieron cargar ambos modelos.")
    exit()

# Número de partidas
num_partidas = int(input("\n¿Cuántas partidas? "))

# Contadores
victorias_j1 = 0
victorias_j2 = 0
empates = 0

# Ejecutar partidas
for i in range(1, num_partidas + 1):
    ganador = jugar_partida(modelo_ia_1, modelo_ia_2, verbose=False)
    
    if ganador == "j1":
        victorias_j1 += 1
    elif ganador == "j2":
        victorias_j2 += 1
    else:
        empates += 1
    
    if i % 10 == 0:
        print(f"[{i}/{num_partidas}] j1: {victorias_j1} | j2: {victorias_j2} | Empates: {empates}")

# Resultados finales
print(f"\n=== RESULTADOS ({num_partidas} partidas) ===")
print(f"IA_Pequeña (j1): {victorias_j1} victorias ({victorias_j1/num_partidas*100:.1f}%)")
print(f"IA_Grande (j2): {victorias_j2} victorias ({victorias_j2/num_partidas*100:.1f}%)")
print(f"Empates: {empates} ({empates/num_partidas*100:.1f}%)")