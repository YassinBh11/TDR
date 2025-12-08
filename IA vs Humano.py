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
def acciones_validas_ia(posiciones, meta_index):
    """Devuelve lista de índices de acciones válidas para J2 (IA)"""
    action_to_move = {
        0: "m1", 1: "m2", 2: "m3",
        3: "p1", 4: "p2", 5: "p3", 6: "p4"
    }

    valid = []
    for act_idx, mov in action_to_move.items():
        if mov.startswith("m"):
            # Muñecos: verificar que no estén en meta
            pieza = f"j2_{mov}"
            if posiciones[pieza] < meta_index:
                valid.append(act_idx)
        else:
            # Pingorote: verificar que haya algún muñeco de J2 en su posición
            pieza = mov
            pos_ping = posiciones.get(pieza)
            if pos_ping is None or pos_ping >= meta_index:
                continue
            
            # Verificar si hay algún muñeco de J2 en esa casilla
            puede_mover = False
            for nm, p in posiciones.items():
                if nm.startswith("j2_m") and p == pos_ping:
                    puede_mover = True
                    break
            
            if puede_mover:
                valid.append(act_idx)
    
    return valid if valid else [0, 1, 2]  # Fallback: muñecos


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


def seleccionar_accion_ia(estado, modelo, posiciones, meta_index):
    """Selecciona la mejor acción válida usando el modelo entrenado"""
    acciones_validas = acciones_validas_ia(posiciones, meta_index)
    
    if not acciones_validas:
        return 0  # Fallback
    
    # Obtener Q-values del modelo
    with torch.no_grad():
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        q_values = modelo(estado_tensor).squeeze(0)
    
    # Filtrar solo acciones válidas y elegir la mejor
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


# ===== CARGAR EL MODELO =====
def cargar_modelo_ia(ruta="modelo_dqn_estrategia_mental.pth"):
    """Carga el modelo entrenado"""
    try:
        checkpoint = torch.load(ruta)
        modelo = Red_neuronal(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size']
        )
        modelo.load_state_dict(checkpoint['model_state_dict'])
        modelo.eval()  # Modo evaluación
        print("✅ Modelo IA cargado exitosamente")
        return modelo
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo del modelo")
        print("   Asegúrate de tener 'modelo_dqn_estrategia_mental.pth' en la carpeta")
        return None
    except Exception as e:
        print(f"ERROR al cargar modelo: {e}")
        return None

titulo= pyfiglet.figlet_format("Estrategia Mental")
print(titulo)

jugador1 = input("Jugador 1, como te llamas?: ")
modo_juego = input("¿Contra quién quieres jugar?\n1. Jugador 2 humano\n2. IA (modelo entrenado)\nElige (1/2): ")

modelo_ia = None
if modo_juego == "2":
    print("\n Cargando IA entrenada...")
    modelo_ia = cargar_modelo_ia("modelo_dqn_estrategia_mental.pth")
    if modelo_ia is None:
        print(" No se pudo cargar la IA. Jugando contra J2 humano.")
        jugador2 = "j2"
    else:
        jugador2 = "IA"
else:
    jugador2 = "j2"

def limpiar():
    os.system("cls" if os.name=="nt" else "clear")
   
def escribir(texto, velocidad=0.04):
    for letra in texto:
        print(letra, end="", flush=True) # el flush=true obliga a python a mostrar las letras directamente y que no guarde en memoria para soltar la frase directamente
        time.sleep(velocidad) #"pausa" entre letra y letra para el efecto de maquina de escribir
    print()

def intro():  # Dejar las instrucciones mucho más claras e ir eliminando el texto

    # Bienvenida
    escribir(f"\nBienvenidos {jugador1} y {jugador2} a Estrategia Mental!")
    time.sleep(1)

    saltar_intro = input("¿Eres nuevo jugador?(Si/No): ")
    if saltar_intro in ("Si", "si"):

        # Información de los creadores y propósito
        limpiar()
        print("\nEste juego ha sido desarrollado por un equipo de tres estudiantes con el objetivo de combinar estrategia y aprendizaje.")
        time.sleep(2.5)
        print("Su creación busca ofrecer una experiencia de pensamiento táctico y análisis durante cada turno, donde cada decisión puede cambiar el curso de la partida.")
        time.sleep(5)

        # Mini historia de lo que va a pasar
        limpiar()
        print("\nEn Estrategia Mental, cada jugador controla tres muñecos que deben avanzar por un tablero lleno de casillas especiales.")
        time.sleep(2.5)
        print("Cada turno implica lanzar un dado y elegir qué muñeco mover, sin saber qué hará el adversario.")
        time.sleep(3)
        print("Tus decisiones influirán directamente en el avance de tus muñecos y en la dinámica del tablero, que puede cambiar con cada movimiento.")
        time.sleep(3)

        escribir("\nPulsa [Enter] para conocer las Reglas del Juego!")
        input()

        # Reglas del juego y condiciones
        limpiar()
        print("Reglas del Juego:")
        time.sleep(1)
        print("1. Cada jugador tiene tres muñecos: m1, m2 y m3.")
        time.sleep(1.5)
        print("2. El tablero tiene 15 casillas: INICIO, casillas positivas, SUERTE, negativas y META.")
        time.sleep(1.5)
        print("3. En cada turno, ambos jugadores lanzan un dado (1-6) y eligen qué muñeco mover.")
        time.sleep(1.5)
        print("4. Las casillas pueden desaparecer si fueron ocupadas y luego quedan vacías.")
        time.sleep(1.5)
        print("5. El juego termina cuando todos los muñecos de ambos jugadores llegan a la META.")
        time.sleep(2)

        escribir("\n¡Pulsa [Enter] para conocer las casillas especiales!")
        input()

        limpiar()
        print("Tipos de Casillas:")
        time.sleep(1)
        print("Casillas Positivas (1-5): avanzan normalmente.")
        print("Casillas SUERTE: pueden tener efectos aleatorios (a definir).")
        print("Casillas Negativas (-5 a -1): pueden penalizar el avance.")
        print("INICIO: punto de partida. META: objetivo final.")
        time.sleep(2)

        escribir("\n¡Pulsa [Enter] para conocer cómo se gana!")
        input()

        limpiar()
        print("\nCondiciones de Victoria:")
        time.sleep(1)
        print("El primer jugador que consiga llevar sus tres muñecos a la casilla META gana automáticamente.")
        print("Si ambos jugadores lo logran en el mismo turno, se declara empate.")
        time.sleep(3)

        print("\nPrepárense para poner a prueba su estrategia y que comience la partida.\n")
        time.sleep(2)
        escribir("Pulse [Enter] para comenzar la partida!")
        input()

    else:
        escribir("\n¡Perfecto! Vayamos directos a la partida!")
        time.sleep(2.5)

puntuacion1= 0
puntuacion2= 0

tablero=[]

inicio= []
tablero.append("INICIO")

casillas_negativas= []
for i in range(-1,-5, -1):   
    tablero.append(i)
   
casillas_suerte= []
for _ in range(2):
    tablero.append("SUERTE")

casillas_positivas= []
for i in range(5,1, -1):
    tablero.append(i)

casillas_negativas= []
for i in range(-1,-6, -1):
    tablero.append(i)

meta=[]
tablero.append("META")

print(tablero)

# Marca si un jugador ya llegó a META
jugador_llego_meta = {"j1": False, "j2": False}

posiciones = {
    "j1_m1": 0,
    "j1_m2": 0,
    "j1_m3": 0,
    "j2_m1": 0,
    "j2_m2": 0,
    "j2_m3": 0,
    "p1": 5,
    "p2": 6,
    "p3": 7,
    "p4": 8,
}

def eleccion_muñecos1():
    if todos_en_meta("j1", posiciones, meta_index):
        return None  # jugador ya no puede mover
    
    opciones = []

    # Añadir solo muñecos de J1 que no estén en META
    for m in ["m1", "m2", "m3"]:
        nombre_muñeco = f"j1_{m}"  # j1_m1, j1_m2, j1_m3
        if posiciones[nombre_muñeco] < meta_index:
            opciones.append(m)  # La opción que ve el jugador sigue siendo "m1", "m2", "m3"
           
    for p in ["p1", "p2", "p3", "p4"]:
        if posiciones[p] < meta_index:
            opciones.append(p)
   
    # Pedir al jugador que elija hasta que sea válido
    while True:
        mov1 = input(f"\n{jugador1}, elige tu movimiento {opciones}: ")
        if mov1 not in opciones:
            print(f"Opción inválida {jugador1}, debe ser una de: {opciones}")
            continue
        break

    return mov1

def eleccion_muñecos2():
    if todos_en_meta("j2", posiciones, meta_index):
        return None  # jugador ya no puede mover
    opciones = []

    # Añadir solo muñecos de J2 que no estén en META
    for m in ["m1", "m2", "m3"]:
        nombre_muñeco = f"j2_{m}"  # j2_m1, j2_m2, j2_m3
        if posiciones[nombre_muñeco] < meta_index:
            opciones.append(m)  # La opción que ve el jugador sigue siendo "m1", "m2", "m3"

    for p in ["p1", "p2", "p3", "p4"]:
        if posiciones[p] < meta_index:
            opciones.append(p)

    # Pedir al jugador que elija hasta que sea válido
    while True:
        mov2 = input(f"\n{jugador2}, elige tu movimiento {opciones}: ")
        if mov2 not in opciones:
            print(f"Opción inválida {jugador2}, debe ser una de: {opciones}")
            continue
        break

    return mov2

def eleccion_muñecos2_ia(posiciones, meta_index, modelo_ia):
    """Versión IA de elección de muñecos para J2"""
    if todos_en_meta("j2", posiciones, meta_index):
        return None
    
    # Obtener estado actual del juego
    estado = obtener_estado_juego(posiciones, tablero)
    
    # La IA elige acción
    accion_idx = seleccionar_accion_ia(estado, modelo_ia, posiciones, meta_index)
    movimiento = traducir_accion_ia(accion_idx)
    print(f"\n IA elige: {movimiento} (acción {accion_idx})")
    
    return movimiento

def poder_mover_pingorote(nombre_pingorote, posiciones):
    # Permite mover el pingorote si hay un muñeco en la misma casilla
    posicion_pingorote = posiciones[nombre_pingorote]
    for nombre, pos in posiciones.items():
        if (("m" in nombre) and pos == posicion_pingorote):
            return True
    return False

def movimientos1(eleccionj1, dadoj1, posiciones):
   
    while True:
        if eleccionj1 == "m1":
            posiciones["j1_m1"] += dadoj1
            break
        elif eleccionj1 == "m2":
            posiciones["j1_m2"] += dadoj1
            break
        elif eleccionj1 == "m3":
            posiciones["j1_m3"] += dadoj1
            break
        elif eleccionj1 in ["p1", "p2", "p3", "p4"]:
            if poder_mover_pingorote(eleccionj1, posiciones):
                posiciones[f"{eleccionj1}"] += dadoj1
                break
            else:
                print(f"\nNo puedes mover {eleccionj1} porque no hay ningún muñeco en su casilla.")
                eleccionj1= eleccion_muñecos1()
        else:
            eleccionj1= eleccion_muñecos1()
    return posiciones

def movimientos2(eleccionj2, dadoj2, posiciones):
   
    while True:
        if eleccionj2 == "m1":
            posiciones["j2_m1"] += dadoj2
            break
        elif eleccionj2 == "m2":
            posiciones["j2_m2"] += dadoj2
            break
        elif eleccionj2 == "m3":
            posiciones["j2_m3"] += dadoj2
            break
        elif eleccionj2 in ["p1", "p2", "p3", "p4"]:
            if poder_mover_pingorote(eleccionj2, posiciones):
                posiciones[f"{eleccionj2}"] += dadoj2
                break
            else:
                print(f"\nNo puedes mover {eleccionj2} porque no hay ningún muñeco en su casilla.")
                eleccionj2= eleccion_muñecos2_ia()
        else:
            eleccionj2= eleccion_muñecos2_ia()
    return posiciones

def mostrar_tablero(tablero, posiciones):
    print("\nEstado del tablero:")
    for i, casilla in enumerate(tablero):
        if casilla== "Vacia":
            continue
       
        contenido = f"Casilla {i}: {casilla}"
       
        # Buscar qué muñecos están en esta casilla
        muñecos_en_casilla = [nombre for nombre, pos in posiciones.items() if pos == i]
       
        if muñecos_en_casilla:
            contenido += " ← " + ", ".join(muñecos_en_casilla)
       
        print(contenido)

# Controlar si un muñeco ha llegado a META
jugador_llego_meta = {nombre: False for nombre in posiciones.keys()}

ultimo_ocupante= {}
casillas_reclamadas= {}

def limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_ocupadas_antes):

    meta_index_anterior = len(tablero) - 1
    posiciones_actuales = set(posiciones.values())

    nuevas_casillas = []
    mapa_indices = {}  # Mapeo: índice antiguo → índice nuevo
    nuevo_indice = 0

    for i, casilla in enumerate(tablero):

        # CASO: eliminar casilla si estuvo ocupada y ahora está vacía (excepto meta)
        if (
            i in casillas_ocupadas_antes
            and i not in posiciones_actuales
            and casilla != "META"
        ):
            propietario = ultimo_ocupante.get(i, None)
            print(f"[TABLERO] Casilla {i} eliminada. Último ocupante: {propietario}")
            continue  # No se copia → se elimina

        # Si NO se elimina, la añadimos
        nuevas_casillas.append(casilla)
        mapa_indices[i] = nuevo_indice
        nuevo_indice += 1

    # Recalcular posiciones según el mapa nuevo
    for nombre in posiciones:
        posiciones[nombre] = mapa_indices[posiciones[nombre]]

    # Recalcular casillas ocupadas antes
    nuevas_ocupadas = {mapa_indices[i] for i in casillas_ocupadas_antes if i in mapa_indices}

    # Recalcular último ocupante con índices nuevos
    nuevo_ultimo = {}
    for i, ocupante in ultimo_ocupante.items():
        if i in mapa_indices:
            nuevo_ultimo[mapa_indices[i]] = ocupante

    # Nuevo meta index
    meta_index = len(nuevas_casillas) - 1

    return nuevas_casillas, nuevas_ocupadas, nuevo_ultimo, meta_index

meta_index= len(tablero)-1
casillas_ocupadas_antes = set(posiciones.values())

orden_llegada = []

def registrar_llegada(nombre):
    """Registra el orden en que los muñecos llegan a META"""
    if nombre not in orden_llegada:
        orden_llegada.append(nombre)

def verificar_meta(posiciones, meta_index):
    for nombre, pos in posiciones.items():
        if nombre.startswith("j"): #SOLO MUÑECOS NO PINGOROTES
            if pos >= meta_index:
                posiciones[nombre] = meta_index  
                registrar_llegada(nombre)
limpiar()
mostrar_tablero(tablero, posiciones)

def todos_en_meta(jugador, posiciones, meta_index):
    return all(posiciones[f"{jugador}_m{i}"] >= meta_index for i in range(1, 4))

def aplicar_puntuacion(jugador, casilla):
    """
    jugador: "j1" o "j2"
    casilla: valor tomado de tablero[índice] (int o "SUERTE")
    Usa globales: puntuacion1, puntuacion2, tablero
    """
    global puntuacion1, puntuacion2, tablero

    # 1) casilla numérica
    if isinstance(casilla, int):
        if jugador == "j1":
            puntuacion1 += casilla
        else:
            puntuacion2 += casilla
        return

    # 2) SUERTE: convertir la casilla negativa más "grande" en positiva
    if casilla == "SUERTE":
        negativos_idx = [i for i, v in enumerate(tablero) if isinstance(v, int) and v < 0]
        if not negativos_idx:
            print(f"{jugador} cayó en SUERTE: no hay casillas negativas para convertir.")
            return
        idx_max_neg = max(negativos_idx, key=lambda i: tablero[i])
        antiguo = tablero[idx_max_neg]
        tablero[idx_max_neg] = abs(tablero[idx_max_neg])
        print(f"{jugador} cayó en SUERTE: la casilla {idx_max_neg} ({antiguo}) se convierte en {tablero[idx_max_neg]}.")
        return

    # 3) INICIO y META → sin efecto
    return

def mostrar_puntuaciones():
    print(f"Puntuación {jugador1}: {puntuacion1}")
    print(f"Puntuación {jugador2}: {puntuacion2}")

# Determinar orden de turno solo en el primer turno
turno= 1
dadoj1 = random.randint(1, 6)
dadoj2 = random.randint(1, 6)

if dadoj1 < dadoj2:
    primero, segundo = jugador1, jugador2
    dado_primero, dado_segundo = dadoj1, dadoj2
elif dadoj2 < dadoj1:
    primero, segundo = jugador2, jugador1
    dado_primero, dado_segundo = dadoj2, dadoj1

else:
    while dadoj1 == dadoj2:
        dadoj1 = random.randint(1, 6)
        dadoj2 = random.randint(1, 6)
    if dadoj1 < dadoj2:
        primero, segundo = jugador1, jugador2
        dado_primero, dado_segundo = dadoj1, dadoj2
    else:
        primero, segundo = jugador2, jugador1
        dado_primero, dado_segundo = dadoj2, dadoj1

if primero == jugador1:
    print(f"\n{jugador1} empieza primero con un {dadoj1} en el dado.")
else:
    print(f"\n{jugador2} empieza primero con un {dadoj2} en el dado.")

# Inventarios para almacenar las casillas reclamadas por cada jugador
inventario_j1 = []
inventario_j2 = []

def mostrar_inventarios():
    print(f"\nInventario de {jugador1}: {inventario_j1}")
    print(f"Inventario de {jugador2}: {inventario_j2}")

# Tokens de SUERTE
tokens_suerte_j1 = 0
tokens_suerte_j2 = 0

def reclamar_casillas(tablero, posiciones, ultimo_ocupante, casillas_reclamadas, inventario_j1, inventario_j2):
    """
    Revisa el tablero y asigna las casillas vacías a los jugadores,
    agregándolas a su inventario.
    """
    ocupadas = set(posiciones.values())
    
    for idx, casilla in enumerate(tablero):
        if idx not in ocupadas and idx not in casillas_reclamadas:
            propietario = ultimo_ocupante.get(idx, None)
            if propietario:
                casillas_reclamadas[idx] = propietario
                if propietario == "j1":
                    inventario_j1.append(casilla)
                else:
                    inventario_j2.append(casilla)
                print(f"Casilla {idx} reclamada por {propietario}")

def marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante):
    pos_nueva = posiciones[pieza_movida]

    # Solo muñecos de jugadores, no pingorotes
    if pieza_movida.startswith("j1_"):
        propietario = "j1"
    elif pieza_movida.startswith("j2_"):
        propietario = "j2"
    else:
        propietario = None

    if propietario is not None:
        ultimo_ocupante[pos_nueva] = propietario

def calcular_puntuacion_final(inventario):
    """
    Convierte negativos en positivos usando las casillas 'SUERTE' reclamadas,
    y devuelve la puntuación final.
    """
    inventario = inventario.copy()
    # Listar todos los valores negativos
    negativos = [v for v in inventario if isinstance(v, int) and v < 0]

    # Por cada casilla SUERTE que exista en el inventario
    for i, valor in enumerate(inventario):
        if valor == "SUERTE" and negativos:
            # Tomar el negativo más grande en valor absoluto (más negativo)
            mayor_negativo = min(negativos)  # más negativo
            idx_neg = inventario.index(mayor_negativo)
            inventario[idx_neg] = abs(mayor_negativo)  # convertir a positivo
            negativos.remove(mayor_negativo)

    # Sumar solo los números para obtener la puntuación
    puntuacion = sum(v for v in inventario if isinstance(v, int))
    return puntuacion

# Bucle de Partida CORREGIDO - Copia y pega directamente
while any(nombre.startswith("j") and pos != meta_index for nombre, pos in posiciones.items()):

    # --- JUEGA EL PRIMERO ---
    if primero == jugador1:
        if not todos_en_meta("j1", posiciones, meta_index):
            if turno == 1:
                dado_actual = dadoj1
            else:
                dadoj1 = random.randint(1, 6)
                print(f"\n{jugador1} ha sacado un {dadoj1} en el dado.")
        else:
            primero, segundo = segundo, primero
            turno += 1
            continue

        eleccionj1 = eleccion_muñecos1()
        if eleccionj1 is None:
            primero, segundo = segundo, primero
            turno += 1
            continue

        pieza_movida = f"j1_{eleccionj1}" if eleccionj1.startswith("m") else eleccionj1
        posicion_origen = posiciones[pieza_movida]  # ← ANTES de mover

        posiciones = movimientos1(eleccionj1, dadoj1, posiciones)
        marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante)

        # Verificar si casilla origen quedó vacía
        ocupantes_origen = [p for p, pos in posiciones.items() if pos == posicion_origen]
        casillas_vacias = set()
        if len(ocupantes_origen) == 0 and posicion_origen != meta_index:
            casillas_vacias.add(posicion_origen)
            
            valor_casilla = tablero[posicion_origen]
            if valor_casilla not in ("INICIO", "META"):
                if pieza_movida.startswith("j1_"):
                    inventario_j1.append(valor_casilla)
                    print(f"{jugador1} reclama casilla {posicion_origen} con valor {valor_casilla}.")
                elif pieza_movida.startswith("j2_"):
                    inventario_j2.append(valor_casilla)
                    print(f"{jugador2} reclama casilla {posicion_origen} con valor {valor_casilla}.")

        # Limitar posiciones a meta
        for nombre in posiciones:
            if nombre.startswith("j") or nombre.startswith("p"):
                posiciones[nombre] = min(posiciones[nombre], meta_index)

        tablero, _, ultimo_ocupante, meta_index = limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_vacias)
        casillas_totales = len(tablero)

        verificar_meta(posiciones, meta_index)
        mostrar_tablero(tablero, posiciones)
        mostrar_inventarios()

    else:  # j2 como primero
        if not todos_en_meta("j2", posiciones, meta_index):
            if turno == 1:
                dado_actual = dadoj2
            else:
                dadoj2 = random.randint(1, 6)
                print(f"\n{jugador2} ha sacado un {dadoj2} en el dado.")
        else:
            primero, segundo = segundo, primero
            turno += 1
            continue

        if modelo_ia is not None:
         eleccionj2 = eleccion_muñecos2_ia(posiciones, meta_index, modelo_ia)
        else:
            eleccionj2 = eleccion_muñecos2()
        if eleccionj2 is None:
            primero, segundo = segundo, primero
            turno += 1
            continue

        pieza_movida = f"j2_{eleccionj2}" if eleccionj2.startswith("m") else eleccionj2
        posicion_origen = posiciones[pieza_movida]  # ← ANTES de mover

        posiciones = movimientos2(eleccionj2, dadoj2, posiciones)
        marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante)

        ocupantes_origen = [p for p, pos in posiciones.items() if pos == posicion_origen]
        casillas_vacias = set()
        if len(ocupantes_origen) == 0 and posicion_origen != meta_index:
            casillas_vacias.add(posicion_origen)
            
            valor_casilla = tablero[posicion_origen]
            if valor_casilla not in ("INICIO", "META"):
                if pieza_movida.startswith("j1_"):
                    inventario_j1.append(valor_casilla)
                    print(f"{jugador1} reclama casilla {posicion_origen} con valor {valor_casilla}.")
                elif pieza_movida.startswith("j2_"):
                    inventario_j2.append(valor_casilla)
                    print(f"{jugador2} reclama casilla {posicion_origen} con valor {valor_casilla}.")

        for nombre in posiciones:
            if nombre.startswith("j") or nombre.startswith("p"):
                posiciones[nombre] = min(posiciones[nombre], meta_index)

        tablero, _, ultimo_ocupante, meta_index = limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_vacias)
        casillas_totales = len(tablero)

        verificar_meta(posiciones, meta_index)
        mostrar_tablero(tablero, posiciones)
        mostrar_inventarios()

    turno += 1

    # --- JUEGA EL SEGUNDO ---
    if segundo == jugador1:
        if not todos_en_meta("j1", posiciones, meta_index):
            dadoj1 = random.randint(1, 6)
            print(f"\n{jugador1} ha sacado un {dadoj1} en el dado.")

            eleccionj1 = eleccion_muñecos1()
            if eleccionj1 is None:
                continue

            pieza_movida = f"j1_{eleccionj1}" if eleccionj1.startswith("m") else eleccionj1
            posicion_origen = posiciones[pieza_movida]

            posiciones = movimientos1(eleccionj1, dadoj1, posiciones)
            marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante)

            ocupantes_origen = [p for p, pos in posiciones.items() if pos == posicion_origen]
            casillas_vacias = set()
            if len(ocupantes_origen) == 0 and posicion_origen != meta_index:
                casillas_vacias.add(posicion_origen)
                
                valor_casilla = tablero[posicion_origen]
                if valor_casilla not in ("INICIO", "META"):
                    if pieza_movida.startswith("j1_"):
                        inventario_j1.append(valor_casilla)
                        print(f"{jugador1} reclama casilla {posicion_origen} con valor {valor_casilla}.")
                    elif pieza_movida.startswith("j2_"):
                        inventario_j2.append(valor_casilla)
                        print(f"{jugador2} reclama casilla {posicion_origen} con valor {valor_casilla}.")

            for nombre in posiciones:
                if nombre.startswith("j") or nombre.startswith("p"):
                    posiciones[nombre] = min(posiciones[nombre], meta_index)

            tablero, _, ultimo_ocupante, meta_index = limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_vacias)
            casillas_totales = len(tablero)

            verificar_meta(posiciones, meta_index)
            mostrar_tablero(tablero, posiciones)
            mostrar_inventarios()

    else:  # j2 como segundo
        if not todos_en_meta("j2", posiciones, meta_index):
            dadoj2 = random.randint(1, 6)
            print(f"\n{jugador2} ha sacado un {dadoj2} en el dado.")

            if modelo_ia is not None:
                eleccionj2 = eleccion_muñecos2_ia(posiciones, meta_index, modelo_ia)
            else:
                eleccionj2 = eleccion_muñecos2()
            if eleccionj2 is None:
                continue

            pieza_movida = f"j2_{eleccionj2}" if eleccionj2.startswith("m") else eleccionj2
            posicion_origen = posiciones[pieza_movida]

            posiciones = movimientos2(eleccionj2, dadoj2, posiciones)
            marcar_ultimo_ocupante(pieza_movida, posiciones, ultimo_ocupante)

            ocupantes_origen = [p for p, pos in posiciones.items() if pos == posicion_origen]
            casillas_vacias = set()
            if len(ocupantes_origen) == 0 and posicion_origen != meta_index:
                casillas_vacias.add(posicion_origen)
                
                valor_casilla = tablero[posicion_origen]
                if valor_casilla not in ("INICIO", "META"):
                    if pieza_movida.startswith("j1_"):
                        inventario_j1.append(valor_casilla)
                        print(f"{jugador1} reclama casilla {posicion_origen} con valor {valor_casilla}.")
                    elif pieza_movida.startswith("j2_"):
                        inventario_j2.append(valor_casilla)
                        print(f"{jugador2} reclama casilla {posicion_origen} con valor {valor_casilla}.")

            for nombre in posiciones:
                if nombre.startswith("j") or nombre.startswith("p"):
                    posiciones[nombre] = min(posiciones[nombre], meta_index)
                    
            tablero, _, ultimo_ocupante, meta_index = limpiar_tablero(tablero, posiciones, ultimo_ocupante, casillas_vacias)
            casillas_totales = len(tablero)

            verificar_meta(posiciones, meta_index)
            mostrar_tablero(tablero, posiciones)
            mostrar_inventarios()

    limpiar()
    mostrar_tablero(tablero, posiciones)
    mostrar_inventarios()

# --- FIN DEL JUEGO ---
#Calcular puntuaciones finales usando inventarios y tokens
puntuacion1 = calcular_puntuacion_final(inventario_j1)
puntuacion2 = calcular_puntuacion_final(inventario_j2)

recompensas = [5, 4, 3, 2, 1, 0]
for i, muñeco in enumerate(orden_llegada):
        puntos = recompensas[min(i, len(recompensas) - 1)]
        if muñeco.startswith("j1_"):
            puntuacion1 += puntos
        elif muñeco.startswith("j2_"):
            puntuacion2 += puntos

print("\n PUNTUACIONES FINALES ")
print(f"{jugador1}: {puntuacion1} puntos")
print(f"{jugador2}: {puntuacion2} puntos")

# Determinar ganador
if puntuacion1 > puntuacion2:
    print(f"{jugador1} ha ganado!")
elif puntuacion2 > puntuacion1:
    print(f"{jugador2} ha ganado!")
else:
    print("¡Empate!")
    print("\n¡El juego ha terminado! Gracias por jugar.") 