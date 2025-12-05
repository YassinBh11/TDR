import random
import time
import os 
import pyfiglet
import colorama
from colorama import Fore, Back, Style, init
init(autoreset=True)

titulo= pyfiglet.figlet_format("MarcosCapi")
print(titulo)

jugador1="j1"
jugador2="j2"

#Añadir colores a las fichas
colores_fichas = {
    "j1": Fore.BLUE + Style.BRIGHT,
    "j2": Fore.RED + Style.BRIGHT,
}

colores_casillas = {
    "INICIO": Back.CYAN + Fore.BLACK,
    "SUERTE": Back.MAGENTA + Fore.WHITE,
    "META": Back.GREEN + Fore.BLACK,
    "VACÍA": Fore.BLACK,
}

def color_casilla(valor):
    if isinstance(valor, int):
        if valor > 0:
            return Fore.GREEN + Style.BRIGHT
        else:
            return Fore.RED
    return colores_casillas.get(valor, Fore.WHITE)
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

intro()  

puntuacion1= 0
puntuacion2= 0

tablero=[]

inicio= []
tablero.append("INICIO")

casillas_positivas= []
for i in range(1,6):
    tablero.append(i)
    
casillas_suerte= []
for _ in range(4):
    tablero.append("SUERTE")

casillas_negativas= []
for i in range(-5,0):
    tablero.append(i)

meta=[]
tablero.append("META")

print(tablero)
casillas_totales= 15

def eleccion_muñecos1():
    m1_jugador1= "m1"
    m2_jugador1= "m2"
    m3_jugador1= "m3"
    pingorote1= "p1"
    pingorote2= "p2"
    pingorote3= "p3"
    pingorote4= "p4"
    
    # Jugador 1
    while True:
        mov1 = input(f"\n{jugador1}, elige tu movimiento ({m1_jugador1}, {m2_jugador1}, {m3_jugador1}, {pingorote1}, {pingorote2}, {pingorote3}, {pingorote4}): ")
        if mov1 not in ["m1", "m2", "m3", "p1", "p2", "p3", "p4"]:
            print(f"Opción inválida {jugador1}, debe ser (m1, m2, m3, p1, p2, p3 o p4)")
            continue
        break        
    return mov1

def eleccion_muñecos2():
    m1_jugador2= "m1"
    m2_jugador2= "m2"
    m3_jugador2= "m3"
    pingorote1= "p1"
    pingorote2= "p2"
    pingorote3= "p3"
    pingorote4= "p4"
    
    # Jugador 2
    while True:
        mov2 = input(f"\n{jugador2}, elige tu movimiento ({m1_jugador2}, {m2_jugador2}, {m3_jugador2}, {pingorote1}, {pingorote2}, {pingorote3}, {pingorote4}): ")
        if mov2 not in ["m1", "m2", "m3", "p1", "p2", "p3", "p4"]:
            print(f"Opción inválida {jugador1}, debe ser (m1, m2, m3, p1, p2, p3 o p4)")
            continue
        break 
            
    return mov2


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
                eleccionj2= eleccion_muñecos2()
        else:
            eleccionj2= eleccion_muñecos2()
    return posiciones

def mostrar_tablero(tablero, posiciones):
    print("\n" + "="*40)
    print(Style.BRIGHT + "      ESTADO DEL TABLERO" + Style.RESET_ALL)
    print("="*40)

    for i, casilla in enumerate(tablero):
        if casilla == "VACÍA":
            color = Fore.BLACK
            texto_casilla = color + str(casilla) + Style.RESET_ALL
        else:
            color = color_casilla(casilla)
            texto_casilla = color + str(casilla) + Style.RESET_ALL

        # Fichas en casilla
        muñecos = [n for n, p in posiciones.items() if p == i]

        if muñecos:
            muñecos_coloreados = ", ".join(
                colores_fichas.get(m[:2], Fore.WHITE + Style.BRIGHT) + m + Style.RESET_ALL for m in muñecos
            )
            print(f"[{i:02}] {texto_casilla} ← {muñecos_coloreados}")
        else:
            print(f"[{i:02}] {texto_casilla}")

    print("="*40)

posiciones = {
    "j1_m1": 0,
    "j1_m2": 0,
    "j1_m3": 0,
    "j2_m1": 0,
    "j2_m2": 0,
    "j2_m3": 0,
    "p1": 6,
    "p2": 7,
    "p3": 8,
    "p4": 9
}

ultimo_ocupante= {}

casillas_reclamadas= {}

def limpiar_tablero(tablero, posiciones, casillas_ocupadas_antes):
    nuevas_casillas = tablero.copy()
    casillas_con_muñecos = set(posiciones.values())
    meta_index = len(tablero) - 1  # define dentro para evitar errores

    for i in range(len(tablero)):
        # Si estaba ocupada antes pero ya no hay muñeco encima, y no es la meta
        if i in casillas_ocupadas_antes and i not in casillas_con_muñecos and i != meta_index:
            print(f"Casilla {i} liberada: {tablero[i]}")
            nuevas_casillas[i] = "VACÍA"
            casillas_reclamadas[i]= ultimo_ocupante[i]
            casillas_ocupadas_antes.remove(i)

    return nuevas_casillas, casillas_ocupadas_antes


meta_index= len(tablero)-1 
casillas_ocupadas_antes = set(posiciones.values())


def verificar_meta(posiciones, meta_index):
    for nombre, pos in posiciones.items():
        if nombre.startswith("j"): #SOLO MUÑECOS NO PINGOROTES 
            if pos >= meta_index:
                posiciones[nombre] = meta_index  
                print(f"{nombre} ha llegado a la meta")

limpiar()
mostrar_tablero(tablero, posiciones)

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


#Bucle de Partida
while any(nombre.startswith("j") and pos !=meta_index for nombre, pos in posiciones.items()):
    
    if primero == jugador1:
        if turno==1:
            dado_actual= dadoj1
        else:
            dadoj1 = random.randint(1, 6)
            dado_actual= dadoj1
            print(f"\n{jugador1} ha sacado un {dadoj1} en el dado.")
        
        eleccionj1= eleccion_muñecos1()
        # Ejecutar movimientos y guardar posiciones
        posiciones_antes = posiciones.copy()
        posiciones = movimientos1(eleccionj1, dadoj1, posiciones)
        
        for nombre, pos in posiciones.items():
            ultimo_ocupante[pos]=nombre[:2]  # Guardar solo j1 o j2
        
        for pos in posiciones.values():
            casillas_ocupadas_antes.add(pos)
        tablero, casillas_ocupadas_antes = limpiar_tablero(tablero, posiciones, casillas_ocupadas_antes)
        casillas_totales=len(tablero)
        verificar_meta(posiciones, meta_index)
        mostrar_tablero(tablero, posiciones)

    else: 
        if turno==1:
            dado_actual= dadoj2
        else:
            dadoj2= random.randint(1,6)
            dado_actual= dadoj2
            print(f"\n{jugador2} ha sacado un {dadoj2} en el dado.")
        
        eleccionj2= eleccion_muñecos2()
        posiciones_antes = posiciones.copy()
        posiciones = movimientos2(eleccionj2, dadoj2, posiciones)
        
        for nombre,pos in posiciones.items():
            ultimo_ocupante[pos]= nombre[:2] 
        
        for pos in posiciones.values():
            casillas_ocupadas_antes.add(pos)
        tablero, casillas_ocupadas_antes = limpiar_tablero(tablero, posiciones, casillas_ocupadas_antes)
        casillas_totales=len(tablero)
        verificar_meta(posiciones, meta_index)
        mostrar_tablero(tablero, posiciones)
    turno+=1
    
    if segundo== jugador1:
        dadoj1 = random.randint(1, 6)
        print(f"\n{jugador1} ha sacado un {dadoj1} en el dado.")
        eleccionj1 = eleccion_muñecos1()
        posiciones_antes = posiciones.copy()
        posiciones = movimientos1(eleccionj1, dadoj1, posiciones)
        
        for nombre,pos in posiciones.items():
            ultimo_ocupante[pos]= nombre[:2] 
        
        for pos in posiciones.values():
            casillas_ocupadas_antes.add(pos)
        tablero, casillas_ocupadas_antes = limpiar_tablero(tablero, posiciones, casillas_ocupadas_antes)
        casillas_totales=len(tablero)
        verificar_meta(posiciones, meta_index)
        mostrar_tablero(tablero, posiciones)

    else:
        dadoj2= random.randint(1,6)
        print(f"\n{jugador2} ha sacado un {dadoj2} en el dado.")
        eleccionj2= eleccion_muñecos2()
        posiciones_antes = posiciones.copy()
        posiciones = movimientos2(eleccionj2, dadoj2, posiciones)

        for nombre,pos in posiciones.items():
            ultimo_ocupante[pos]= nombre[:2] 

        for pos in posiciones.values():
            casillas_ocupadas_antes.add(pos)
        tablero, casillas_ocupadas_antes = limpiar_tablero(tablero, posiciones, casillas_ocupadas_antes)
        casillas_totales=len(tablero)
        verificar_meta(posiciones, meta_index)
        mostrar_tablero(tablero, posiciones)
    turno+=1
    limpiar()
    mostrar_tablero(tablero, posiciones)

tablero, casillas_ocupadas_antes = limpiar_tablero(tablero, posiciones, casillas_ocupadas_antes)
casillas_totales = len(tablero)

puntuaciones_casillas=  {}
for i, casilla in enumerate(tablero):
        if isinstance(casilla, int):
            puntuaciones_casillas[i] = casilla
        elif casilla== "SUERTE":
            puntuaciones_casillas[i] = 0
        elif casilla== "INICIO":
            puntuaciones_casillas[i] = 0
        elif casilla== "META":
            puntuaciones_casillas[i] = 0

for i, casilla in enumerate(tablero):
    if casilla== "SUERTE" and i in casillas_reclamadas:
        max_neg= max([val for val in puntuaciones_casillas.values() if val <0], default=None)
        for idx, val in puntuaciones_casillas.items():
            if val== max_neg:
                puntuaciones_casillas[idx]= abs(val)
                print(f"La casilla {idx} se ha vuelto positiva gracias a la Suerte!")
                break
            
puntuacion1 = sum(
    puntuaciones_casillas[i]
    for i, j in casillas_reclamadas.items()
    if j == "j1" and i in puntuaciones_casillas
)

puntuacion2 = sum(
    puntuaciones_casillas[i]
    for i, j in casillas_reclamadas.items()
    if j == "j2" and i in puntuaciones_casillas
)

print("\n --- Resultado Final ---")
print("\nCasillas reclamadas por cada jugador:")
print(f"\n{jugador1} se quedó con las casillas:")
for i, jugador in casillas_reclamadas.items():
    if jugador == "j1":
        print(f"  - Casilla {i}: {puntuaciones_casillas[i]} puntos")

print(f"\n{jugador2} se quedó con las casillas:")
for i, jugador in casillas_reclamadas.items():
    if jugador == "j2":
        print(f"  - Casilla {i}: {puntuaciones_casillas[i]} puntos")
