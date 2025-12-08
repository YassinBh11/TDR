import gymnasium as gym
from gymnasium import spaces
import numpy as np #Biblioteca para calculos complejos y aplicacion de formulas
import random

import torch #Red Neuronal
import torch.nn as nn
import torch.optim as optim #Optimizador errores para predicciones futuras precisas

from collections import deque 
import time

# Hiperparámetros DQN
batch_size = 64          # Cantidad de experiencias que tomamos de la memoria para entrenar
gamma = 0.99             # Factor de descuento de recompensas futuras
epsilon_start = 1.0      # Probabilidad inicial de explorar
epsilon_end = 0.05       # Probabilidad mínima de explorar
epsilon_decay = 0.995    # Factor de decaimiento por episodioº  
memory_capacity = 10000  # Tamaño de la memoria de experiencias
num_episodes = 100         # Número de partidas/episodios de entrenamiento

class Juego(gym.Env):

    def __init__(self):
        self.puntuacion1 = 0
        self.puntuacion2 = 0
        self.ultimo_ocupante = {}
        self.casillas_reclamadas = {}
        self.inventario_j1 = []
        self.inventario_j2 = []
        self.tokens_suerte_j1 = 0
        self.tokens_suerte_j2 = 0
        self.done = False
        self.tablero = []
        self.posiciones = {}
        self.casillas_ocupadas_antes = set()
        self.orden_llegada = []
        self.meta_index = 0
        self.action_space = spaces.Discrete(7) 
        self.observation_space = spaces.Box(
            low=0,
            high=35,
            shape=(11,),
            dtype=np.int32
        )
        self.reset()

    def reset(self):
        self.tablero = []
        self.inicio= []
        self.tablero.append("INICIO")

        self.casillas_negativas= []
        for i in range(-1,-5, -1):   
            self.tablero.append(i)
   
        self.casillas_suerte= []
        for _ in range(2):
            self.tablero.append("SUERTE")

        casillas_positivas= []
        for i in range(5,1, -1):
            self.tablero.append(i)

        casillas_negativas= []
        for i in range(-1,-6, -1):
            self.tablero.append(i)

        self.meta=[]
        self.tablero.append("META")
        self.meta_index = len(self.tablero) - 1

        self.posiciones = {
            "j1_m1": 0, "j1_m2": 0, "j1_m3": 0,
            "j2_m1": 0, "j2_m2": 0, "j2_m3": 0,
            "p1": 5, "p2": 6, "p3": 7, "p4": 8
        }
        
        self.jugador_llego_meta = {nombre: False for nombre in self.posiciones.keys()}
        self.ultimo_ocupante= {}
        self.casillas_reclamadas= {}
        self.casillas_ocupadas_antes = set(self.posiciones.values())
        self.orden_llegada = []
        self.inventario_j1 = []
        self.inventario_j2 = []

        self.tokens_suerte_j1 = 0
        self.tokens_suerte_j2 = 0

        self.turno= 1
        self.dado1 = random.randint(1, 6)
        self.dado2 = random.randint(1, 6)

        self.first_turn = True
        self.done = False

        return self.get_obs(), {}

    def get_obs(self):
        return np.array([
            self.posiciones["j1_m1"],
            self.posiciones["j1_m2"],
            self.posiciones["j1_m3"],
            self.posiciones["j2_m1"],
            self.posiciones["j2_m2"],
            self.posiciones["j2_m3"],
            self.posiciones["p1"],
            self.posiciones["p2"],
            self.posiciones["p3"],
            self.posiciones["p4"],
            len(self.tablero),
        ], dtype=np.int32)

    def limpiar_tablero(self):
        self.meta_index_anterior = len(self.tablero) - 1
        posiciones_actuales = set(self.posiciones.values())

        nuevas_casillas = []
        mapa_indices = {}
        nuevo_indice = 0

        for i, casilla in enumerate(self.tablero):
            if (
                i in self.casillas_ocupadas_antes
                and i not in posiciones_actuales
                and casilla != "META"
            ):
                propietario = self.ultimo_ocupante.get(i, None)
                continue

            nuevas_casillas.append(casilla)
            mapa_indices[i] = nuevo_indice
            nuevo_indice += 1

        for nombre in self.posiciones:
            self.posiciones[nombre] = mapa_indices[self.posiciones[nombre]]

        nuevas_ocupadas = {mapa_indices[i] for i in self.casillas_ocupadas_antes if i in mapa_indices}

        nuevo_ultimo = {}
        for i, ocupante in self.ultimo_ocupante.items():
            if i in mapa_indices:
                nuevo_ultimo[mapa_indices[i]] = ocupante

        self.meta_index = len(nuevas_casillas) - 1

        return nuevas_casillas, nuevas_ocupadas, nuevo_ultimo, self.meta_index

    def registrar_llegada(self, nombre):
        if nombre not in self.orden_llegada:
            self.orden_llegada.append(nombre)

    def verificar_meta(self):
        for nombre, pos in self.posiciones.items():
            if pos >= self.meta_index:
                    self.posiciones[nombre] = self.meta_index  
                    self.registrar_llegada(nombre)

    def todos_en_meta(self, jugador):
        return all(self.posiciones[f"{jugador}_m{i}"] >= self.meta_index for i in range(1, 4))

    def aplicar_puntuacion(self, jugador, casilla):
        if isinstance(casilla, int):
            if jugador == "j1":
                self.puntuacion1 += casilla
            else:
                self.puntuacion2 += casilla
            return

        if casilla == "SUERTE":
            negativos_idx = [i for i, v in enumerate(self.tablero) if isinstance(v, int) and v < 0]
            if not negativos_idx:
                print(f"{jugador} cayó en SUERTE: no hay casillas negativas para convertir.")
                return
            idx_max_neg = max(negativos_idx, key=lambda i: self.tablero[i])
            antiguo = self.tablero[idx_max_neg]
            self.tablero[idx_max_neg] = abs(self.tablero[idx_max_neg])
            print(f"{jugador} cayó en SUERTE: la casilla {idx_max_neg} ({antiguo}) se convierte en {self.tablero[idx_max_neg]}.")
            return

        return

    def reclamar_casillas(self):
        self.ocupadas = set(self.posiciones.values())
        
        for idx, casilla in enumerate(self.tablero):
            if idx not in self.ocupadas and idx not in self.casillas_reclamadas:
                propietario = self.ultimo_ocupante.get(idx, None)
                if propietario:
                    self.casillas_reclamadas[idx] = propietario
                    if propietario == "j1":
                        self.inventario_j1.append(casilla)
                    else:
                        self.inventario_j2.append(casilla)

    def marcar_ultimo_ocupante(self, pieza_movida):
        self.posiciones_nueva = self.posiciones[pieza_movida]

        if pieza_movida.startswith("j1_"):
            propietario = "j1"
        elif pieza_movida.startswith("j2_"):
            propietario = "j2"
        else:
            propietario = None

        if propietario is not None:
            self.ultimo_ocupante[self.posiciones_nueva] = propietario

    def calcular_puntuacion_final(self, inventario):
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

    def acciones_validas(self, jugador):
        """Devuelve lista de índices de acciones válidas para el jugador"""
        action_to_move = {
            0: "m1", 1: "m2", 2: "m3",
            3: "p1", 4: "p2", 5: "p3", 6: "p4"
        }

        valid = []
        for act_idx, mov in action_to_move.items():
            if mov.startswith("m"):
                # Siempre se pueden mover muñecos
                valid.append(act_idx)
            else:
                # Pingorote: verificar si hay algún muñeco en su posición
                pieza = mov
                pos_ping = self.posiciones.get(pieza)
                if pos_ping is None:
                    continue
                
                # Verificar si hay algún muñeco del jugador en esa casilla
                puede_mover = False
                for nm, p in self.posiciones.items():
                    if nm.startswith(f"{jugador}_m") and p == pos_ping:
                        puede_mover = True
                        break
                
                if puede_mover:
                    valid.append(act_idx)
        
        return valid
        
    def step(self, action):
        if self.done:
            return self.get_obs(), 0.0, True, False, {}

        action_to_move = {
                0: "m1", 1: "m2", 2: "m3",
                3: "p1", 4: "p2", 5: "p3", 6: "p4"
            }
        
        def _poder_mover_pingorote(ping_name):
            pos_ping = self.posiciones[ping_name]
            for nm, p in self.posiciones.items():
                if ("_m" in nm) and p == pos_ping:
                    return True
            return False

        # helper: ejecutar una acción numérica para un jugador dado (j1/j2)
        def _ejecutar_accion_jugador(jugador, act_idx, dado):
            mov = action_to_move.get(act_idx, None)
            if mov is None:
                # acción inválida → no mover
                return None, None, -100

            # traducir a nombre de pieza según jugador
            if mov.startswith("m"):
                pieza = f"{jugador}_{mov}"  # ej. j1_m1
            else:
                pieza = mov  # p1..p4 (pingorotes son globales)

            # Si es pingorote, comprobar permiso
            if pieza in ("p1","p2","p3","p4"):
                if not _poder_mover_pingorote(pieza):
                    # pingorote no se puede mover -> no mover
                    return None, None, -100
                
            # origen antes de mover
            origen = self.posiciones.get(pieza, None)
            if origen is None:
                # pieza inexistente (defensa) -> no mover
                return None, None, -100

            # aplicar movimiento
            self.posiciones[pieza] += dado

            # limitar a meta
            if self.posiciones[pieza] >= self.meta_index:
                self.posiciones[pieza] = self.meta_index

            # marcar último ocupante de la nueva posición (se hará con la función de la clase si existe)
            # guardamos pieza movida para posible uso por otras funciones
            return pieza, origen, 0.0  # no reward por mover

        # Determinar orden primera vez
        if getattr(self, "first_turn", True):
            d1 = getattr(self, "dado1", random.randint(1, 6))
            d2 = getattr(self, "dado2", random.randint(1, 6))

            while d1 == d2:
                d1 = random.randint(1, 6)
                d2 = random.randint(1, 6)

            if d1 > d2:
                self.primero = "j1"
                self.segundo = "j2"
            else:
                self.primero = "j2"
                self.segundo = "j1"

            self._first_d1 = d1
            self._first_d2 = d2
            self.first_turn = True

        info = {"moves": []}

        # para cada jugador en el orden [primero, segundo]
        for turno_jugador in (self.primero, self.segundo):
            # elegir dado: si es el primer turno, usar los dados guardados; si no, tirar
            if getattr(self, "first_turn", False):
                # usar dados guardados dependientes de jugador
                if turno_jugador == "j1":
                    dado = getattr(self, "_first_d1", random.randint(1, 6))
                else:
                    dado = getattr(self, "_first_d2", random.randint(1, 6))
            else:
                dado = random.randint(1, 6)

            # decidir acción a ejecutar
            if turno_jugador == "j1":
                # la acción del agente viene del argumento `action`
                act_idx = int(action)
            else:
                # rival: acción válida aleatoria
                acciones_validas_rival = self.acciones_validas("j2")
                if acciones_validas_rival:
                    act_idx = random.choice(acciones_validas_rival)
                else:
                    act_idx = 0  # Mover primer muñeco por defecto

            # ejecutar acción
            pieza_movida, posicion_origen, reward = _ejecutar_accion_jugador(turno_jugador, act_idx, dado)

            if pieza_movida is not None:
                try:
                    self.marcar_ultimo_ocupante(pieza_movida)
                except Exception:
                    pos_nueva = self.posiciones[pieza_movida]
                    propietario = "j1" if pieza_movida.startswith("j1_") else ("j2" if pieza_movida.startswith("j2_") else None)
                    if propietario is not None:
                        self.ultimo_ocupante[pos_nueva] = propietario

                if posicion_origen is not None:
                    ocupantes_en_origen = [n for n, p in self.posiciones.items() if p == posicion_origen]
                    if len(ocupantes_en_origen) == 0 and posicion_origen != self.meta_index:
                        if posicion_origen not in self.casillas_reclamadas:
                            valor = self.tablero[posicion_origen]
                            if valor not in ("INICIO", "META"):
                                owner_inv = self.inventario_j1 if pieza_movida.startswith("j1_") else self.inventario_j2
                                owner_inv.append(valor)
                                self.casillas_reclamadas[posicion_origen] = "j1" if pieza_movida.startswith("j1_") else "j2"

            info["moves"].append({"player": turno_jugador, "action": act_idx, "piece": pieza_movida, "die": dado})

        if getattr(self, "first_turn", False):
            self.first_turn = False

        for nombre in list(self.posiciones.keys()):
            if nombre.startswith("j") or nombre.startswith("p"):
                self.posiciones[nombre] = min(self.posiciones[nombre], self.meta_index)

        try:
            nuevas, nuevas_ocupadas, nuevo_ultimo, nuevo_meta = self.limpiar_tablero()
            self.tablero = nuevas
            self.casillas_ocupadas_antes = nuevas_ocupadas
            self.ultimo_ocupante = nuevo_ultimo
            self.meta_index = nuevo_meta
        except Exception:
            pass

        try:
            self.reclamar_casillas()
        except Exception:
            pass

        try:
            self.verificar_meta()
        except Exception:
            for nombre, pos in self.posiciones.items():
                if nombre.startswith("j") and pos >= self.meta_index:
                    if nombre not in getattr(self, "orden_llegada", []):
                        try:
                            self.registrar_llegada(nombre)
                        except Exception:
                            self.orden_llegada.append(nombre)

        j1_win = self.todos_en_meta("j1")
        j2_win = self.todos_en_meta("j2")

        if j1_win or j2_win:
            puntuacion_j1 = self.calcular_puntuacion_final(self.inventario_j1)
            puntuacion_j2 = self.calcular_puntuacion_final(self.inventario_j2)
            
            recompensas = [5, 4, 3, 2, 1, 0]
            for i, muñeco in enumerate(self.orden_llegada):
                puntos = recompensas[min(i, len(recompensas) - 1)]
                if muñeco.startswith("j1_"):
                    puntuacion_j1 += puntos
                elif muñeco.startswith("j2_"):
                    puntuacion_j2 += puntos
            
            diferencia = puntuacion_j1 - puntuacion_j2
            
            if diferencia >= 0:
                reward = puntuacion_j1 ** 2
            else:
                reward = -1 * (diferencia ** 2)

            self.done = True
            obs = self.get_obs()
            return obs, float(reward), self.done, False, info        
        obs = self.get_obs()
        return obs, 0.0, False, False, info 


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


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory) 
    

#Entrenamiento IA
def seleccionar_accion(state, model, epsilon, env):
    state_tensor = torch.FloatTensor(state).unsqueeze(0) 
    
    acciones_validas = env.acciones_validas("j1")
    
    # Si no hay acciones válidas, mover primer muñeco por defecto
    if not acciones_validas:
        return 0
    
    if random.random() < epsilon:
        return random.choice(acciones_validas)  # Exploración: acción válida aleatoria
    else:
        # with torch.no_grad() indica que no vamos a calcular gradientes
        # Solo queremos obtener el valor Q, no entrenar todavía
        
        with torch.no_grad():  
            # Pasamos el estado por la red y obtenemos los Q-values
            q_values = model(state_tensor).squeeze(0) #Q-values para cada acción
        
        # Obtener Q-values solo de acciones válidas y elegir la mejor
        q_validas = [(i, q_values[i].item()) for i in acciones_validas]
        mejor_accion = max(q_validas, key=lambda x: x[1])[0]
         
        # torch.argmax(q_values) devuelve el índice de la acción con mayor Q-value
        return mejor_accion  # .item() convierte el tensor en un número entero normal


model = Red_neuronal()
optimizer = optim.Adam(model.parameters(), lr=0.001)
memory = ReplayMemory(memory_capacity)
criterion = nn.MSELoss()
epsilon = epsilon_start


def entrenar():
    global epsilon  # Para actualizar epsilon
    model.train()
    
    for episodio in range(num_episodes):
        env = Juego()
        estado = env.reset()
        done = False
        total_reward = 0
        ronda = 0
        
        while not done:
            # Extraer array de observación
            estado_array = estado[0] if isinstance(estado, tuple) else estado
            
            # Seleccionar acción
            accion = seleccionar_accion(estado_array, model, epsilon, env)
            ronda += 1
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, info = env.step(accion)
            next_state_array = next_state[0] if isinstance(next_state, tuple) else next_state
            done = terminated or truncated
            
            # Guardar experiencia
            memory.push(estado_array, accion, reward, next_state_array, done)
            
            # Actualizar estado
            estado = next_state
            total_reward += reward
            
            # Debug (opcional, comenta para acelerar)
            print(f"Episodio {episodio+1}, Ronda {ronda}: Acción={accion}, Reward={reward:.1f}")
        
        # Decaimiento epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Entrenamiento si hay suficientes experiencias
        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states, dtype=np.float32))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32))
            dones = torch.FloatTensor(dones)
            
            # Calcular Q values actuales
            q_values = model(states).gather(1, actions).squeeze()
            
            # Calcular Q targets
            with torch.no_grad():
                next_q_values = model(next_states).max(1)[0]
                q_targets = rewards + gamma * next_q_values * (1 - dones)
            
            # Backpropagation
            loss = criterion(q_values, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Episodio {episodio+1}: Loss={loss.item():.4f}, Epsilon={epsilon:.3f}")
        
        if episodio % 100 == 0:
            print(f"Episodio {episodio+1}/{num_episodes} completado, Total Reward: {total_reward}")


def evaluar(modelo, num_partidas=1):
    env = Juego()
    victorias = 0
    empates = 0
    derrotas = 0
    
    for i in range(num_partidas):
        estado = env.reset()
        done = False
        episodio_reward = 0
        
        while not done:
            estado_array = estado[0] if isinstance(estado, tuple) else estado
            accion = seleccionar_accion(estado_array, modelo, epsilon=0.0, env=env)
            estado, reward, terminated, truncated, info = env.step(accion)
            done = terminated or truncated
            episodio_reward += reward
        
        if episodio_reward > 0:
            victorias += 1
        elif episodio_reward == 0:
            empates += 1
        else:
            derrotas += 1
    
    print(f"Evaluación ({num_partidas} partidas): {victorias}V, {empates}E, {derrotas}D")
    win_rate = victorias / num_partidas * 100
    print(f"Tasa de victorias: {win_rate:.1f}%")

inicio = time.time()
entrenar()
evaluar(model)
print(f"Tiempo total entrenamiento: {time.time() - inicio:.2f} segundos")

torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': 11,
    'hidden_size': 64,
    'output_size': 7
}, "modelo_dqn_estrategia_mental.pth")