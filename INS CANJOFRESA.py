import random
import time
import pyfiglet

titulo= pyfiglet.figlet_format("Estrategia Mental") #PYFIGLET es una librería que transforma texto en letras grandes de consola, para poder hacer ASCII art
print(titulo)

jugador1=input("Jugador 1: Como te llamas? ") 
jugador2=input("Jugador 2: Como te llamas? ")

def escribir(texto, velocidad=0.04):
    for letra in texto:
        print(letra, end="", flush=True) # el flush=true obliga a python a mostrar las letras directamente y que no guarde en memoria para soltar la frase directamente 
        time.sleep(velocidad) #"pausa" entre letra y letra para el efecto de maquina de escribir
    print()

def intro():
    # Bienvenida
    escribir(f"\nBienvenidos {jugador1} y {jugador2} a Estrategia Mental!")
    time.sleep(1)
    
    saltar_intro=input("¿Eres nuevo jugador?(Si/No): ")
    if saltar_intro in ("Si", "si"):
        
    # Información de los creadores y propósito
        print("\nEste juego ha sido desarrollado por un equipo de tres estudiantes con el objetivo de combinar estrategia y aprendizaje.")
        time.sleep(2.5)
        print("Su creación busca ofrecer una experiencia de pensamiento táctico y análisis durante cada turno, donde cada decisión puede cambiar el curso de la partida.")
        time.sleep(3)

    # Mini historia de lo que va a pasar
        print("\nEn Estrategia Mental, cada jugador asumirá el rol de un estratega enfrentándose a su oponente en una serie de turnos.")
        time.sleep(2.5)
        print("Cada turno implica lanzar un dado y elegir cuidadosamente una acción estratégica, sin conocer la elección del adversario.")
        time.sleep(3)
        print("Tus decisiones influirán directamente en tus puntos y en los resultados de tu contrincante, generando un duelo de astucia y previsión.")
        time.sleep(3)

        escribir("\nPulsa [Enter] para conocer las Reglas del Juego!")
        input()

    # Reglas del juego y condiciones
        print("Reglas del Juego:")
        time.sleep(1)
        print("1. Cada turno, ambos jugadores lanzan un dado de seis caras. Los resultados son visibles para los dos jugadores.")
        time.sleep(2)
        print("2. Después de ver los resultados, cada jugador selecciona una opción estratégica: Precisión, Contraataque o Sabotaje.")
        time.sleep(2)
        print("3. No se puede repetir la misma opción dos turnos consecutivos.")
        time.sleep(1.5)

        escribir("\n¡Ahora que ya conocéis las reglas del juego, pulsa [Enter] para conocer las Opciones Estratégicas que tendréis a vuestra disposición!")
        input()

        print("Opciones Estratégicas:")
        time.sleep(1)
        print("Precisión (A): se toma el valor real del dado. Es efectiva contra Contraataque, pero pierde frente a Sabotaje.")
        print("  - Si el oponente elige Sabotaje, el valor se reduce a 1.")
        time.sleep(1.5)

        escribir("\n¡Pulsa [Enter] para conocer el siguiente movimiento!")
        input()

        print("\nContraataque (B): se toma 7 menos el valor del dado. Es efectiva contra Sabotaje, pero pierde frente a Precisión.")
        print("  - Si ambos eligen Contraataque, no ocurre ningún efecto.")
        print("  - Si el oponente elige Precisión, el resultado se reduce a 1.")
        print("  - Si el oponente elige Sabotaje, se obtiene un bono de 4 puntos.")
        time.sleep(1.5)

        escribir("\n¡Pulsa [Enter] para conocer el último movimiento!")
        input()

        print("\nSabotaje (O): ignora el valor propio del dado. Es efectiva contra Precisión, pero pierde frente a Contraataque.")
        print("  - Si el oponente elige Precisión, su valor se reduce a 1 y obtienes los puntos que él habría sumado.")
        print("  - Si el oponente elige Contraataque, obtienes 1 punto adicional.")
        print("  - Si ambos eligen Sabotaje, no ocurre ningún efecto.")
        time.sleep(1.5)

        escribir("\nAhora que ya conocéis todo sobre el juego, os preguntaréis: Y como podemos ganar? ¡Pulsa [Enter] para resolver vuestra duda!")
        input()

        print("\nCondiciones de Victoria:")
        time.sleep(1)
        print("El primer jugador que alcance o supere 21 puntos gana automáticamente.")
        print("Si ambos jugadores alcanzan 21 o más puntos en el mismo turno y terminan con la misma puntuación, el resultado será empate.")
        time.sleep(3)

        print("\nPrepárense para poner a prueba su estrategia y que comience la partida.\n")
        time.sleep(2)
        escribir("Pulse [Enter] para comenzar la partida!")
        input()
#Opcional para el juego:
#elif saltar_intro not in ("Si", "si", "No", "no"):
#print("Respuesta inválida, por favor responde Si o No")
#intro()
    else:
        escribir("\n¡Perfecto! Vayamos directos a la partida!")
intro() #Llamamos a la función intro para que se ejecute y se vea en pantalla

puntuacion1=0
puntuacion2=0 

ultima_eleccion1= None
ultima_eleccion2= None

#Eleccion movimiento

def eleccion_ABO(ultima_eleccion1, ultima_eleccion2,):
        # Jugador 1
    while True:
        mov1 = input(f"\n{jugador1}, elige tu movimiento (A, B u O): ")
        if mov1 not in ("A", "B", "O"):
            print(f"Opción inválida {jugador1}, debe ser A, B u O")
            continue
        if mov1 == ultima_eleccion1:
            print(f"\n{jugador1}, no puedes repetir el mismo movimiento dos turnos seguidos.")
            continue
        break  # si está bien, salimos del bucle

    # Jugador 2
    while True:
        mov2 = input(f"{jugador2}, elige tu movimiento (A, B u O): ")
        if mov2 not in ("A", "B", "O"):
            print(f"Opción inválida {jugador2}, debe ser A, B u O")
            continue
        if mov2 == ultima_eleccion2:
            print(f"\n{jugador2}, no puedes repetir el mismo movimiento dos turnos seguidos.")
            continue
        break  # si está bien, salimos del bucle
            
    return mov1, mov2

def movimientos(eleccionj1, eleccionj2, dadoj1, dadoj2):
    puntos=0 
    
    #Movimiento A
    if eleccionj1== "A":
        puntos = dadoj1
    
        if eleccionj2== "O":
            puntos = 1
    
        if eleccionj2== "A":
            pass
    
        if eleccionj2== "B":
            pass
   #Movimiento B
    
    if eleccionj1== "B":
        puntos = 7-dadoj1
        
        if eleccionj2== "A":
            puntos = 1

        if eleccionj2 == "O":
            puntos = 11-dadoj1

        if eleccionj2 == "B":
            pass
    
    #Movimiento O
    
    if eleccionj1== "O":
        dadoj1=0
    
        if eleccionj2== "A":
            puntos=dadoj2
    
        if eleccionj2== "B":
            puntos+= 1
    
    return puntos 

#Bucle de Partida
ronda=1
while puntuacion1 < 21 or puntuacion2 < 21:
    
    escribir(f"\n--- Ronda {ronda} ---")
    time.sleep(0.5)
    
    dado1= random.randint(1,6)
    dado2= random.randint(1,6)

    print(f"\n{jugador1} tu dado ha sacado {dado1}")
    print(f"{jugador2} tu dado ha sacado {dado2}")

    mov1, mov2 = eleccion_ABO(ultima_eleccion1, ultima_eleccion2)
    ultima_eleccion1, ultima_eleccion2= mov1, mov2

    puntos1 = movimientos(mov1, mov2, dado1, dado2)
    puntos2 = movimientos(mov2, mov1, dado2, dado1)

    puntuacion1 += puntos1
    puntuacion2 += puntos2

    ronda += 1 # aumentar la ronda al final de cada turno

    print(f"\n{jugador1} gana {puntos1} puntos (total: {puntuacion1})")
    print(f"{jugador2} gana {puntos2} puntos (total: {puntuacion2})")
    time.sleep(2)

#Sistema de Ganador Final con Empate
    if puntuacion1 >= 21 and puntuacion2 >= 21:
        print("\n¡Empate! Ambos jugadores habeis acabado con la misma puntuación")
        break
    elif puntuacion1 >= 21:
        print(f"\n ¡{jugador1} gana la partida con {puntuacion1} puntos!")
        break
    elif puntuacion2 >= 21:   
        print(f"\n ¡{jugador2} gana la partida con {puntuacion2} puntos!") 
        break
