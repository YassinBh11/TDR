import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque 
import time

# Hiperparámetros DQN
batch_size = 64
gamma = 0.999
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.9995
memory_capacity = 100000
num_episodes = 1000

class Juego(gym.Env):

    def __init__(self):
        self.puntuacion1 = 0
        self.puntuacion2 = 0
        self.victorias_j1 = 0
        self.victorias_j2 = 0
        self.empates = 0
        self.last_result= None
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

        num_suerte_en_inventario = sum(1 for v in inventario if v == "SUERTE")
        
        if num_suerte_en_inventario > 0 and negativos:
            # Ordenar negativos por más negativo (ej: -7, -5, -3, -1)
            negativos_ordenados = sorted([v for v in inventario if isinstance(v, int) and v < 0])
            # Convertir hasta num_suerte_en_inventario de los más negativos
            for _ in range(min(num_suerte_en_inventario, len(negativos_ordenados))):
                peor = negativos_ordenados.pop(0)  # sacamos el más negativo
                idx_neg = inventario.index(peor)
                inventario[idx_neg] = abs(peor)

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
                valid.append(act_idx)
            else:
                pieza = mov
                pos_ping = self.posiciones.get(pieza)
                if pos_ping is None:
                    continue
                
                puede_mover = False
                for nm, p in self.posiciones.items():
                    if nm.startswith(f"{jugador}_m") and p == pos_ping:
                        puede_mover = True
                        break
                
                if puede_mover:
                    valid.append(act_idx)
        
        return valid
    
    def obtener_negativos_en_inventario(self, jugador):
        """Devuelve lista de valores negativos en el inventario del jugador"""
        inventario = self.inventario_j1 if jugador == "j1" else self.inventario_j2
        return [v for v in inventario if isinstance(v, int) and v < 0]

    def calcular_reward_casilla(self, jugador, casilla_reclamada):
        
        if casilla_reclamada == "INICIO" or casilla_reclamada == "META":
            return 0.0

        tiene_suerte = (self.tokens_suerte_j1 > 0) if jugador == "j1" else (self.tokens_suerte_j2 > 0)
        negativos = self.obtener_negativos_en_inventario(jugador)

        # ------------------------------------------------------------
        # 1. CASILLA SUERTE — MUY PRIORITARIA
        # ------------------------------------------------------------
        if casilla_reclamada == "SUERTE":
            reward_base = 2500.0

            if negativos:
                mayor_neg = min(negativos)
                valor_abs = abs(mayor_neg)

                if valor_abs >= 4:
                    return reward_base + valor_abs * 150.0  # combo fuerte
                else:
                    return reward_base - 500.0  # algo menos valiosa si negativos pequeños
            
            return reward_base


        # ------------------------------------------------------------
        # 2. CASILLA NUMÉRICA
        # ------------------------------------------------------------
        if isinstance(casilla_reclamada, int):

            # -------------------------
            # POSITIVAS
            # -------------------------
            if casilla_reclamada > 0:
                return float(casilla_reclamada) * 1200.0


            # -------------------------
            # NEGATIVAS
            # -------------------------
            valor_abs = abs(casilla_reclamada)

            if tiene_suerte:
                # Convertir un negativo sigue siendo malo,
                # pero menos porque lo transformará luego.
                return -valor_abs * 100.0   # pequeña penalización

            else:
                # Sin SUERTE: NEGATIVOS PROHIBIDOS
                if valor_abs <= 3:
                    return -5000.0  # castigo brutal

                if valor_abs >= 4:
                    # grande pero sin SUERTE → sigue siendo malísimo
                    return -10000.0 
                return -15000.0  # castigo máximo

        return 0.0
        
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

        def _ejecutar_accion_jugador(jugador, act_idx, dado):
            mov = action_to_move.get(act_idx, None)
            if mov is None:
                return None, None, -1.0 # Retornamos un float negativo por defecto

            if mov.startswith("m"):
                pieza = f"{jugador}_{mov}"
            else:
                pieza = mov

            if pieza in ("p1","p2","p3","p4"):
                if not _poder_mover_pingorote(pieza):
                    return None, None, -1.0
                
            origen = self.posiciones.get(pieza, None)
            if origen is None:
                return None, None, -1.0

            # Movimiento real
            self.posiciones[pieza] += dado

            if self.posiciones[pieza] >= self.meta_index:
                self.posiciones[pieza] = self.meta_index

            return pieza, origen, 0.0

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
        reward_acumulado_j1 = 0.0

        # Para cada jugador en el orden [primero, segundo]
        for turno_jugador in (self.primero, self.segundo):
            if getattr(self, "first_turn", False):
                if turno_jugador == "j1":
                    dado = getattr(self, "_first_d1", random.randint(1, 6))
                else:
                    dado = getattr(self, "_first_d2", random.randint(1, 6))
            else:
                dado = random.randint(1, 6)

            if turno_jugador == "j1":
                act_idx = int(action)
            else:
                acciones_validas_rival = self.acciones_validas("j2")
                if acciones_validas_rival:
                    act_idx = random.choice(acciones_validas_rival)
                else:
                    act_idx = 0

            # Guardar estado ANTES de la acción
            inventario_anterior = self.inventario_j1.copy() if turno_jugador == "j1" else self.inventario_j2.copy()
            tokens_suerte_anterior = self.tokens_suerte_j1 if turno_jugador == "j1" else self.tokens_suerte_j2
            
            # --- ARREGLO DE ERROR Y LÓGICA DE RECOMPENSAS ---
            # 1. Asignamos el retorno a reward_movimiento directamente
            pieza_movida, posicion_origen, reward_movimiento = _ejecutar_accion_jugador(turno_jugador, act_idx, dado)

            if pieza_movida is not None:
                # 2. Reward Shaping: Recompensa por avanzar (base)
                reward_movimiento += float(dado) * 50.0
                
                # 3. ANTI-TUNNEL VISION: Recompensa extra por sacar ficha de INICIO
                if posicion_origen == 0:
                    reward_movimiento += 300.0
                
                # Penalización pequeña si intenta mover algo que ya está en meta
                if posicion_origen == self.meta_index:
                    reward_movimiento -= 50.0

                try:
                    self.marcar_ultimo_ocupante(pieza_movida)
                except Exception:
                    pos_nueva = self.posiciones[pieza_movida]
                    propietario = "j1" if pieza_movida.startswith("j1_") else ("j2" if pieza_movida.startswith("j2_") else None)
                    if propietario is not None:
                        self.ultimo_ocupante[pos_nueva] = propietario

                # Reclamar casilla si se dejó vacía
                if posicion_origen is not None:
                    ocupantes_en_origen = [n for n, p in self.posiciones.items() if p == posicion_origen]
                    if len(ocupantes_en_origen) == 0 and posicion_origen != self.meta_index:
                        if posicion_origen not in self.casillas_reclamadas:
                            valor_casilla = self.tablero[posicion_origen]
                            if valor_casilla not in ("INICIO", "META"):
                                if turno_jugador == "j1":
                                    self.inventario_j1.append(valor_casilla)
                                    if valor_casilla == "SUERTE":
                                        self.tokens_suerte_j1 += 1
                                else:
                                    self.inventario_j2.append(valor_casilla)
                                    if valor_casilla == "SUERTE":
                                        self.tokens_suerte_j2 += 1
                                
                                self.casillas_reclamadas[posicion_origen] = turno_jugador
            else:
                # Si el movimiento fue inválido (pieza_movida is None)
                if turno_jugador == "j1":
                    reward_movimiento = -100.0  # Castigo

            # Detectar qué casilla se reclamó para sumar su puntuación
            if turno_jugador == "j1":
                inventario_actual = self.inventario_j1
            else:
                inventario_actual = self.inventario_j2
            
            casilla_reclamada = None
            if len(inventario_actual) > len(inventario_anterior):
                casilla_reclamada = inventario_actual[-1]
            
            reward_casilla = 0.0
            if casilla_reclamada is not None:
                # Lógica temporal para calcular reward correcto
                if turno_jugador == "j1":
                    self.tokens_suerte_j1 = tokens_suerte_anterior
                else:
                    self.tokens_suerte_j2 = tokens_suerte_anterior
                
                inventario_actual.pop()
                reward_casilla = self.calcular_reward_casilla(turno_jugador, casilla_reclamada)
                inventario_actual.append(casilla_reclamada)
                
                if turno_jugador == "j1":
                    if casilla_reclamada == "SUERTE":
                        self.tokens_suerte_j1 = tokens_suerte_anterior + 1
                else:
                    if casilla_reclamada == "SUERTE":
                        self.tokens_suerte_j2 = tokens_suerte_anterior + 1
            
            # --- SUMA FINAL DE RECOMPENSAS J1 ---
            if turno_jugador == "j1":
                # Sumamos el reward por moverse + el reward por lo que haya cogido
                reward_acumulado_j1 += (reward_movimiento + reward_casilla)

            info["moves"].append({
                "player": turno_jugador, 
                "action": act_idx, 
                "piece": pieza_movida, 
                "die": dado,
                "casilla_reclamada": casilla_reclamada,
                "reward_turno": (reward_movimiento + reward_casilla) if turno_jugador == "j1" else 0
            })

        if getattr(self, "first_turn", False):
            self.first_turn = False

        # Limitar posiciones a meta
        for nombre in list(self.posiciones.keys()):
            if nombre.startswith("j") or nombre.startswith("p"):
                self.posiciones[nombre] = min(self.posiciones[nombre], self.meta_index)

        # Limpieza y lógica de juego estándar
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
            # Fallback de verificación de meta
            for nombre, pos in self.posiciones.items():
                if nombre.startswith("j") and pos >= self.meta_index:
                    if nombre not in getattr(self, "orden_llegada", []):
                        try:
                            self.registrar_llegada(nombre)
                        except Exception:
                            self.orden_llegada.append(nombre)

        j1_win = self.todos_en_meta("j1")
        j2_win = self.todos_en_meta("j2")

        reward_final = reward_acumulado_j1

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
            print(f"Puntuaciones finales - J1: {puntuacion_j1}, J2: {puntuacion_j2}, Dif: {diferencia}")

            if diferencia > 0:
                self.last_result = "j1"
                self.victorias_j1 += 1
                reward_final += 10000.0 + (diferencia * 100) # Gran bonus por ganar
            elif diferencia < 0:
                self.last_result = "j2"
                self.victorias_j2 += 1
                reward_final -= 5000.0 # Castigo por perder
            else:
                self.last_result = "empate"
                self.empates += 1
                
            print(f"Resultado final de la partida: {self.last_result.upper()}")
            
            self.done = True
            return self.get_obs(), float(reward_final), self.done, False, info

        return self.get_obs(), float(reward_final), False, False, info

class Red_neuronal(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, output_size=7):
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
    

def seleccionar_accion(state, model, epsilon, env):
    state_tensor = torch.FloatTensor(state).unsqueeze(0) 
    
    acciones_validas = env.acciones_validas("j1")
    
    if not acciones_validas:
        return 0
    
    if random.random() < epsilon:
        return random.choice(acciones_validas)
    else:
        with torch.no_grad():  
            q_values = model(state_tensor).squeeze(0)
        
        q_validas = [(i, q_values[i].item()) for i in acciones_validas]
        mejor_accion = max(q_validas, key=lambda x: x[1])[0]
         
        return mejor_accion


model = Red_neuronal()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
memory = ReplayMemory(memory_capacity)
criterion = nn.MSELoss()
epsilon = epsilon_start

def entrenar():
    global epsilon
    model.train()
    
    total_victorias_j1 = 0
    total_victorias_j2 = 0
    total_empates = 0
    
    for episodio in range(num_episodes):
        env = Juego()
        estado = env.reset()
        done = False
        total_reward = 0
        ronda = 0
        
        while not done:
            estado_array = estado[0] if isinstance(estado, tuple) else estado
            
            accion = seleccionar_accion(estado_array, model, epsilon, env)
            ronda += 1
            
            next_state, reward, terminated, truncated, info = env.step(accion)
            next_state_array = next_state[0] if isinstance(next_state, tuple) else next_state
            done = terminated or truncated
            
            memory.push(estado_array, accion, reward, next_state_array, done)
            
            estado = next_state
            total_reward += reward
            
            # Debug mejorado
            if "moves" in info and len(info["moves"]) > 0:
                for move in info["moves"]:
                    if move["player"] == "j1" and move.get("casilla_reclamada"):
                        cas = move["casilla_reclamada"]
                        rew = move.get("reward_turno", 0)
                        neg_count = len([x for x in env.inventario_j1 if isinstance(x, int) and x < 0])
                        sue_count = env.tokens_suerte_j1
                        print(f"Ep{episodio+1} R{ronda}: J1 reclamó {cas} → Reward={rew:.1f} | Inv: {neg_count} neg, {sue_count} suerte")
        
        total_victorias_j1 += env.victorias_j1
        total_victorias_j2 += env.victorias_j2
        total_empates += env.empates
        print(f"Fin Episodio {episodio+1}: Ganador -> {env.last_result.upper()}")
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states, dtype=np.float32))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32))
            dones = torch.FloatTensor(dones)
            
            q_values = model(states).gather(1, actions).squeeze()
            
            with torch.no_grad():
                next_q_values = model(next_states).max(1)[0]
                q_targets = rewards + gamma * next_q_values * (1 - dones)
            
            loss = criterion(q_values, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if episodio % 1 == 0:
                print(f"Episodio {episodio+1}: Loss={loss.item():.4f}, Epsilon={epsilon:.3f}")
        
        if episodio % 1 == 0:
            print(f"Episodio {episodio+1}/{num_episodes} completado, Total Reward: {total_reward:.2f}")
    print("-" * 30)
    print("RESUMEN FINAL DEL ENTRENAMIENTO")
    print(f"Total Episodios: {num_episodes}")
    print(f"Victorias J1 (IA): {total_victorias_j1} ({(total_victorias_j1/num_episodes)*100:.1f}%)")
    print(f"Empates: {total_empates}")
    print("-" * 30)
entrenar()
# Guardar el modelo
print("\nGuardando modelo...")
checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': 11,
        'hidden_size': 128,
        'output_size': 7
    }

torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': 11,
    'hidden_size': 128,
    'output_size': 7
}, "modelo_dqn_estrategia_mental_1000.pth")