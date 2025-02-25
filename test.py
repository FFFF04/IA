#pipe.py: Template para implementação do projeto de Inteligência Artificial 2023/2024.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 31:
# 106067 Francisco Morão
# 106154 Pedro Leal

from sys import stdin
import numpy as np
import copy
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)
class PipeManiaState:
    state_id = 0
    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe
    def diminui(self):
        PipeManiaState.state_id -= 1
    def aumenta(self):
        self.state_id += 1
    def diminui_self(self):
        self.state_id -= 1
    def set(self,value: int):
        self.state_id = value
    
    def confirma_rotacao(self, i: int, j: int, value: str) -> bool:
        n = self.board.n
        vertical_neighbours = self.board.adjacent_vertical_values(i,j)
        above_value = vertical_neighbours[0]
        below_value = vertical_neighbours[1]
        horizontal_neighbours = self.board.adjacent_horizontal_values(i,j)
        left_value = horizontal_neighbours[0]
        right_value = horizontal_neighbours[1]
        diagonal_neighbours = self.board.adjacent_diagonal_values(i,j)

        if i == 0 and j == 0: #Canto superior esquerdo
            if value == "FB" and below_value[0] == "F":
                return False
            if value == "FD" and right_value[0] == "F":
                return False
            if value not in ("FB","VB","FD"):
                return False       

        elif i == 0 and j == n - 1: #Canto superior direito
            if value == "FB" and below_value[0] == "F":
                return False
            if value == "FE" and left_value[0] == "F":
                return False
            if value not in ("FB","VE","FE"):
                return False
            if left_value in ("BC","BB","VB","LH","FD","BD","VD") and value in ("FB"):
                return False
            elif left_value in ("FC","VC","FB","FE","BE","VE","LV") and value in ("FE","VE"):
                return False
            
        elif i == n - 1 and j == 0: #Canto inferior esquerdo
            if value == "FC" and above_value[0] == "F":
                return False
            if value == "FD" and right_value[0] == "F":
                return False
            if value not in ("FC","VD","FD"):
                return False  
            if above_value in ("FC","BC","VC","LH","FE","FD","VD") and value in ("FC","VD"):
                return False
            elif above_value in ("FB","BB","VB","BE","VE","LV","BD") and value in ("FD"):
                return False
            
        elif i == n - 1 and j == n - 1: #Canto inferior direito
            if value == "FC" and above_value[0] == "F":
                return False
            if value == "FE" and left_value[0] == "F":
                return False
            if value not in ("FC","VC","FE"):
                return False
            if above_value in ("FC","BC","VC","LH","FE","FD","VD") and value in ("FC","VC"):
                return False
            elif above_value in ("FB","BB","VB","BE","VE","LV","BD") and value in ("FE"):
                return False

        elif j != 0 and j != n - 1 and (i == 0 or i == n - 1): # Extremidade mais acima ou abaixo   
            if i == 0: # Extremidade mais acima
                if value in ("FC","BC","VC","LV","BE","VD","BD"):
                    return False
            elif i == n - 1: # Extremidade mais abaixo
                if value in ("FB","BB","VB","LV","VE","BE","BD"):
                    return False 
                
            if left_value in ("BC","BB","VB","LH","FD","BD","VD") and value in ("FC","FB","VB","LV","FD","BD","VD"):
                return False
            elif left_value in ("FC","VC","FB","FE","BE","VE","LV") and value in ("BC","VC","BB","LH","FE","BE","VE"):
                return False
            
            if j == n - 2 and i == n - 1:
                if value in ("BC","BB","VB","LH","FD","BD","VD") and right_value in ("FC","FB","VB","LV","FD","BD","VD"):
                    return False
                elif value in ("FC","VC","FB","FE","BE","VE","LV") and right_value in ("BC","VC","BB","LH","FE","BE","VE"):
                    return False

        elif (j == 0 or j == n - 1) and i != 0 and i != n - 1: # Extremidade mais à esquerda ou à direita
            if j == 0 : # Extremidade mais à esquerda
                if value in ("BC","VC","BB","LH","FE","BE","VE"):
                    return False
            elif j == n - 1: # Extremidade mais à direita
                if value in ("BC","BB","VB","LH","FD","BD","VD"):
                    return False

            if above_value in ("FC","BC","VC","LH","FE","FD","VD") and value in ("FC","BC","VC","LV","BE","BD","VD"):
                return False
            elif above_value in ("FB","BB","VB","BE","VE","LV","BD") and value in ("FB","BB","VB","LH","FE","VE","FD"):
                return False

        else:
            if above_value in ("FC","BC","VC","LH","FE","FD","VD") and value in ("FC","BC","VC","LV","BE","BD","VD"):
                return False
            elif above_value in ("FB","BB","VB","BE","VE","LV","BD") and value in ("FB","BB","VB","LH","FE","VE","FD"):
                return False

            if left_value in ("BC","BB","VB","LH","FD","BD","VD") and value in ("FC","FB","VB","LV","FD","BD","VD"):
                return False
            elif left_value in ("FC","VC","FB","FE","BE","VE","LV") and value in ("BC","VC","BB","LH","FE","BE","VE"):
                return False
            
            if j == n - 2:
                if value in ("BC","BB","VB","LH","FD","BD","VD") and right_value in ("FC","FB","VB","LV","FD","BD","VD"):
                    return False
                elif value in ("FC","VC","FB","FE","BE","VE","LV") and right_value in ("BC","VC","BB","LH","FE","BE","VE"):
                    return False
            
            if i == n - 2:
                if value in ("FC","BC","VC","LH","FE","FD","VD") and below_value in ("FC","BC","VC","LV","BE","BD","VD"):
                    return False
                elif value in ("FB","BB","VB","BE","VE","LV","BD") and below_value in ("FB","BB","VB","LH","FE","VE","FD"):
                    return False
                
        if value[0] == "F":
            if value == "FB" and i != n - 1 and below_value[0] == "F":
                return False
            elif value == "FD" and j != n - 1 and right_value[0] == "F":
                return False
            elif value == "FE" and j != 0 and left_value[0] == "F":
                return False
            elif value == "FC" and i != 0 and above_value[0] == "F":
                return False
            
        elif value[0] == "B": 
            if value == "BC" and i != 0 and j != n - 1 and j != 0 and left_value[0] == right_value[0] == above_value[0] == "F":
                return False
            elif value == "BB" and i != n - 1 and j != n - 1 and j != 0 and left_value[0] == right_value[0] == below_value[0] == "F":
                return False
            elif value == "BE" and i != n - 1 and i != 0 and j != 0 and left_value[0] == below_value[0] == above_value[0] == "F":
                return False
            elif value == "BD" and i != n - 1 and j != n - 1 and i != 0 and below_value[0] == right_value[0] == above_value[0] == "F":
                return False
        
        elif value[0] == "V" and n > 2:
            if value == "VC" and i != 0 and j != 0 and left_value[0] == above_value[0] == "F":
                return False
            elif value == "VB" and i != n - 1 and j != n - 1 and right_value[0] == below_value[0] == "F":
                return False
            elif value == "VE" and i != n - 1 and j != 0 and left_value[0] == below_value[0] == "F":
                return False
            elif value == "VD" and i != 0 and j != n - 1 and right_value[0] == above_value[0] == "F":
                return False
            elif value == "VC" and i != 0 and j != 0 and above_value == "VE" and left_value == "VD" and diagonal_neighbours[0] == "VB":
                return False 
            
        elif value[0] == "L":
            if value == "LH" and j != 0 and j != n - 1 and left_value[0] == right_value[0] == "F":
                return False
            elif value == "LV" and i != 0 and i != n - 1 and above_value[0] == below_value[0] == "F":
                return False
        return True   

    def confirma_90(self, i: int, j: int) -> bool:
        piece = self.board.matrix[i][j]
        piece_id = piece[0]
        piece_orientation = piece[1]
        
        if piece_id != "L":
            if piece_orientation == "C":
                value = "".join([piece_id, "D"])

            elif piece_orientation == "B":
                value = "".join([piece_id, "E"])

            elif piece_orientation == "E":
                value = "".join([piece_id, "C"])

            elif piece_orientation == "D":
                value = "".join([piece_id, "B"])          
        if piece_id == "L":
            if piece_orientation == "H":
                value = "".join([piece_id, "V"])

            elif piece_orientation == "V":
                value = "".join([piece_id, "H"])
            
        if not self.confirma_rotacao(i,j,value):
            return False
            
        return True

    def confirma_90_anti(self, i: int, j: int) -> bool:
        piece = self.board.matrix[i][j]
        piece_id = piece[0]
        piece_orientation = piece[1]
        
        if piece_id != "L":
            if piece_orientation == "C":
                value = "".join([piece_id, "E"])
                        
            elif piece_orientation == "B":
                value = "".join([piece_id, "D"])
                        
            elif piece_orientation == "E":
                value = "".join([piece_id, "B"])

            elif piece_orientation == "D":
                value = "".join([piece_id, "C"])
                        
        elif piece_id == "L":
            return False
                
        if not self.confirma_rotacao(i,j,value):
            return False
            
        return True

    def confirma_180(self, i: int, j: int) -> bool:
        piece = self.board.matrix[i][j]
        piece_id = piece[0]
        piece_orientation = piece[1]
        
        if piece_id != "L":
            if piece_orientation == "C":
                value = "".join([piece_id, "B"])
                    
            elif piece_orientation == "B":
                value = "".join([piece_id, "C"])
                        
            elif piece_orientation == "E":
                value = "".join([piece_id, "D"])
                    
            elif piece_orientation == "D":
                value = "".join([piece_id, "E"])  
                        
        elif piece_id == "L":
            return False
        
        if not self.confirma_rotacao(i,j,value):
            return False
            
        return True
    
    def confirma_0(self, i: int, j: int) -> bool:
        value = self.board.matrix[i][j]
                
        if not self.confirma_rotacao(i,j,value):
            return False
            
        return True
    
    def confirma_multipla_rotacao(self,i,j) -> bool:
        value = self.board.matrix[i][j]
        n = self.board.n
        list_action = []
        contador = 0
        
        # vertical_neighbours = self.board.adjacent_vertical_values(i,j)
        # above_value = vertical_neighbours[0]
        # if i > 0 and j != 0 and j != n - 1 and value[0] == "L" and above_value:
        #     self.aumenta()
        #     return list_action
        # first_value = ""
        while True:
            value = self.board.matrix[i][j] 
            if value[0] == "L" and (i == 0 or i == n - 1): 
                if value[1] == "H": #Horizontal
                    list_action.append((i, j, True, True))  # no move
                    contador += 1
                    j += 1
                    if j >= n - 1:
                        break
                        
                elif value[1] == "V": #Vertical
                    # value = "".join([value[0], "H"])
                    list_action.append((i, j, True, False))
                    contador += 1
                    j += 1
                    if j >= n - 1:
                        break
                        
            elif value[0] == "L" and (j == 0 or j == n - 1):
                if value[1] == "V":
                    list_action.append((i, j, True, True))  # no move
                    contador += 1
                    i += 1
                    if i >= n - 1:
                        break
                elif value[1] == "H":
                    # value = "".join([value[0], "V"])
                    list_action.append((i, j, True, False))
                    contador += 1
                    i += 1
                    if i >= n - 1:
                        break

            # elif value[0] == "L" and i != 0 and i != n - 1 and j != 0 and j != n - 1:
            #     if value[1] == "H": #Horizontal
            #         if first_value == "" and self.confirma_rotacao(i,j,value):
            #             first_value = "H"
            #             list_action.append((i, j, True, True))  # no move
            #             j += 1
            #             if j >= n - 1:
            #                 break
            #         elif first_value == "H":
            #             list_action.append((i, j, True, True))
            #             j += 1
            #             if j >= n - 1:
            #                 break
            #         else:
            #             list_action.append((i, j, True, False))
            #             break
            #     elif value[1] == "V": #Vertical
            #         value = "".join([value[0], "H"])
            #         if first_value == "" and self.confirma_rotacao(i,j,value):
            #             list_action.append((i, j, True, False))
            #             first_value = "H"
            #             j += 1
            #             if j >= n - 1:
            #                 break
            #         elif first_value == "H":
            #             list_action.append((i, j, True, False))
            #             j += 1
            #             if j >= n - 1:
            #                 break
            #         else:
            #             list_action.append((i, j, True, True))
            #             break
                
            elif value[0] == "B" and (i == 0 or i == n - 1):
                if (value[1] == "B" and i == 0) or (value[1] == "C" and i == n-1):
                    list_action.append((i, j, True, True))  # no move
                if (value[1] == "D" and i == 0) or (value[1] == "E" and i == n-1):
                    list_action.append((i, j, True, False))  # 90 degrees clockwise
                if (value[1] == "E" and i == 0) or (value[1] == "D" and i == n-1):
                    list_action.append((i, j, False, False))  # 90 degrees anti-clockwise
                if (value[1] == "C" and i == 0) or (value[1] == "B" and i == n-1):
                    list_action.append((i, j, False, True))  # 180 degrees
                contador += 1
                j += 1
                if j >= n - 1:
                    break

            elif value[0] == "B" and (j == 0 or j == n - 1):
                if (value[1] == "D" and j == 0) or (value[1] == "E" and j == n-1):
                    list_action.append((i, j, True, True))  # no move
                if (value[1] == "C" and j == 0) or (value[1] == "B" and j == n-1):
                    list_action.append((i, j, True, False))  # 90 degrees clockwise
                if (value[1] == "B" and j == 0) or (value[1] == "C" and j == n-1):
                    list_action.append((i, j, False, False))  # 90 degrees anti-clockwise
                if (value[1] == "E" and j == 0) or (value[1] == "D" and j == n-1):
                    list_action.append((i, j, False, True))  # 180 degrees
                contador += 1
                i += 1
                if i >= n - 1:
                    break
            else:
                break
        if contador > 1:
            self.set(self.state_id + contador - 1)
        return list_action

class Board:
    """ Representação interna de uma grelha de PipeMania. """
    
    def __init__(self, n: int, matrix: np.array):
        self.n = n
        self.matrix = matrix
        
    def adjacent_vertical_values(self, row: int, col: int) -> tuple:
        """ Devolve os valores imediatamente acima e abaixo,
        respectivamente. """
        
        if row == 0:
            above = None
            below = self.matrix[1][col] 
        elif row == self.n - 1:
            above = self.matrix[self.n - 2][col]
            below = None
        else:
            above = self.matrix[row-1][col]
            below = self.matrix[row+1][col]
    
        return (above, below)
    
    def adjacent_horizontal_values(self, row: int, col: int) -> tuple:
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        
        if col == 0:
            left = None
            right = self.matrix[row][1] 
        elif col == self.n - 1:
            left = self.matrix[row][self.n-2]
            right = None
        else:
            left = self.matrix[row][col-1]
            right = self.matrix[row][col+1]
    
        return (left, right)
    
    def adjacent_diagonal_values(self, row: int, col: int) -> tuple:
        upper_left = " "
        upper_right = " "
        lower_left = " "
        lower_right = " "
        if col == 0:
            upper_left = None
            lower_left = None
           
        elif col == self.n - 1:
            upper_right = None
            lower_right = None
            
        if row == 0:
            upper_left = None
            upper_right = None
            
        elif row == self.n - 1:
            lower_left = None
            lower_right = None
            
        if upper_left != None:
            upper_left = self.matrix[row-1][col-1]
            
        if upper_right != None:
            upper_right = self.matrix[row-1][col+1]
            
        if lower_left != None:    
            lower_left = self.matrix[row+1][col-1]
            
        if lower_right != None:
            lower_right = self.matrix[row+1][col+1]
    
        return (upper_left, upper_right, lower_left, lower_right)
    
    def get_value(self, row: int, col: int) -> str:
       """ Retorna o valor preenchido numa determinada posição """
       return self.matrix[row][col]
   
    def print(self):
        """ Imprime a grelha no formato descrito na secção 4.2."""
        for row in self.matrix:
            print("\t".join(row))
        return ""
       
    
    @staticmethod
    def parse_instance():
        """
        Lê a instância do problema do standard input (stdin)
        e retorna uma instância da classe Board.
        """
        """matrix = np.loadtxt(stdin, dtype=str)
        n = matrix.shape[0]
        board = Board(n, matrix)
        return board"""

        matrix_aux = []
        n = 0
        line = stdin.readline().rstrip().split()
        
        while line != []:
            matrix_aux.append(line)
            n += 1
            line = stdin.readline().rstrip().split()
            
        matrix = np.array(matrix_aux)
        board = Board(n, matrix)
        return board
  
class PipeMania(Problem):
    def __init__(self, initial_state: Board):
        """ O construtor especifica o estado inicial. """
       
        self.initial = initial_state

    def actions(self, state: PipeManiaState) -> list:
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """
        if isinstance(state,PipeManiaState) == False:
            state = PipeManiaState(state)
            state.diminui()
        
        state.aumenta()
        n = state.board.n
        list_of_actions = []
        state_id = state.state_id
        i, j = 0, 0
        if state_id > n**2:
            return list_of_actions
        if 0 < state_id <= n: # extremidade superior
            i = 0
            j = state_id - 1
        elif n < state_id <= 2*n - 1: # extremidade esquerda
            i = state_id - n
            j = 0
        elif 2*n - 1 < state_id <= 3*n - 2: # extremidade direita
            i = state_id - (2*n - 1)
            j = n - 1
        elif 3*n - 2 < state_id <= 4*n - 4: # extremidade inferior
            i = n - 1
            j = state_id - (3*n - 2)  
        else:
            state_id_value = state_id - 4*n+4
            i = ((state_id_value - 1) // (n-2)) + 1
            j = (state_id_value)- ((i-1)*(n-2))

        

        value = state.board.matrix[i][j]
        # (value[0] == "B" and (i == 0 or i == n - 1 or j == 0 or j == n - 1))
        #  if (value[0] == "L" and (i == 0 or i == n - 1)) or (value[0] == "L" and (j == 0 or j == n - 1)):
        if (value[0] == "L" and (i == 0 or i == n - 1)) or (value[0] == "L" and (j == 0 or j == n - 1)):
            multipla_rotacao = state.confirma_multipla_rotacao(i,j)
            list_of_actions.append(multipla_rotacao)
            return list_of_actions

        if state.confirma_0(i, j):
            list_of_actions.append((i, j, True, True))  # no move
        if state.confirma_90(i, j):
            list_of_actions.append((i, j, True, False))  # 90 degrees clockwise
        if state.confirma_90_anti(i, j):
            list_of_actions.append((i, j, False, False))  # 90 degrees anti-clockwise
        if state.confirma_180(i, j):
            list_of_actions.append((i, j, False, True))  # 180 degrees
                
        return list_of_actions
    
    def rotate_piece(self, new_state: PipeManiaState, action: tuple):
        row = action[0]
        col = action[1]
        piece = new_state.board.matrix[row][col]
        piece_id = piece[0]
        piece_orientation = piece[1]
        
        matrix = new_state.board.matrix
        
        if action[2] == True and action[3] == True:
            return new_state

        if piece_id == "F" or piece_id == "B" or piece_id == "V":
            if piece_orientation == "C":
                if action[2] == True:
                    matrix[row][col] = "".join([piece_id, "D"])
                elif action[2] == False and action[3] == False:
                    matrix[row][col] = "".join([piece_id, "E"])
                elif action[2] == False and action[3] == True:
                    matrix[row][col] = "".join([piece_id, "B"])
                    
            elif piece_orientation == "B":
                if action[2] == True:
                    matrix[row][col] = "".join([piece_id, "E"])
                elif action[2] == False and action[3] == False:
                    matrix[row][col] = "".join([piece_id, "D"])
                elif action[2] == False and action[3] == True:
                    matrix[row][col] = "".join([piece_id, "C"])
                        
            elif piece_orientation == "E":
                if action[2] == True:
                    matrix[row][col] = "".join([piece_id, "C"])
                elif action[2] == False and action[3] == False:
                    matrix[row][col] = "".join([piece_id, "B"])
                elif action[2] == False and action[3] == True:
                    matrix[row][col] = "".join([piece_id, "D"])
                    
            elif piece_orientation == "D":
                if action[2] == True:
                    matrix[row][col] = "".join([piece_id, "B"])
                elif action[2] == False and action[3] == False:
                    matrix[row][col] = "".join([piece_id, "C"])
                elif action[2] == False and action[3] == True:
                    matrix[row][col] = "".join([piece_id, "E"])  
                        
        elif piece_id == "L":
            if piece_orientation == "H" and action[3] == False:
                matrix[row][col] = "".join([piece_id, "V"])
            elif piece_orientation == "V" and action[3] == False:
                matrix[row][col] = "".join([piece_id, "H"])
        return new_state

    def result(self, state: PipeManiaState, action) -> PipeManiaState:
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). """
        
        if isinstance(state, PipeManiaState) == False:
            state = PipeManiaState(state)
            state.diminui()

        n = state.board.n
        new_matrix = copy.deepcopy(state.board.matrix)
        board = Board(n,new_matrix)
        new_state_id = copy.copy(state.state_id)
        new_state = PipeManiaState(board)
        new_state.set(new_state_id)

        if isinstance(action, tuple) == False:
            while action != []:
                action_tuple = action.pop(0)
                self.rotate_piece(new_state,action_tuple)
        else:
            self.rotate_piece(new_state,action)
           
        return new_state
    
    def goal_test(self, state: PipeManiaState) -> bool:
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        if isinstance(state, PipeManiaState) == False:
            state = PipeManiaState(state)
            state.diminui()
        
        n = state.board.n
        matrix = state.board.matrix
        
        if state.state_id != n**2:
            return False
        
        matrix_h = [[0 for _ in range(board.n)] for _ in range(board.n)]        
        count = 0
        stack = [(0,0)]
        numeric_value = matrix_h[0][0] + 1

        while stack != []:
            value = stack.pop(0)          
            matrix_h[value[0]][value[1]] = numeric_value
            count += 1

            vertical = state.board.adjacent_vertical_values(value[0],value[1])
            horizontal = state.board.adjacent_horizontal_values(value[0],value[1])
            above_value = vertical[0]
            below_value = vertical[1]
            left_value = horizontal[0]
            right_value = horizontal[1]

            if above_value != None and matrix_h[value[0]-1][value[1]] != numeric_value: # above
                if above_value in ("FB","BB","VB","BE","VE","LV","BD") and matrix[value[0]][value[1]] in ("FC","BC","VC","BE","VD","LV","BD"):
                    stack.append((value[0]-1,value[1]))
                    matrix_h[value[0]-1][value[1]] = numeric_value

            if below_value != None and matrix_h[value[0]+1][value[1]] != numeric_value: # below
                if matrix[value[0]][value[1]] in ("FB","BB","VB","BE","VE","LV","BD") and below_value in ("FC","BC","VC","BE","VD","LV","BD"):
                    stack.append((value[0]+1,value[1]))
                    matrix_h[value[0]+1][value[1]] = numeric_value

            if left_value != None and matrix_h[value[0]][value[1]-1] != numeric_value: # left
                if left_value in ("BC","BB","VB","LH","FD","BD","VD") and matrix[value[0]][value[1]] in ("BC","VC","BB","LH","FE","BE","VE"):
                    stack.append((value[0],value[1]-1))
                    matrix_h[value[0]][value[1]-1] = numeric_value

            if right_value != None and matrix_h[value[0]][value[1]+1] != numeric_value: # right
                if matrix[value[0]][value[1]] in ("BC","BB","VB","LH","FD","BD","VD") and right_value in ("BC","VC","BB","LH","FE","BE","VE"):
                    stack.append((value[0],value[1]+1))
                    matrix_h[value[0]][value[1]+1] = numeric_value

        if count < n**2:
            return False

        return True
        
         
    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        pass
    
if __name__ == "__main__":
    board = Board.parse_instance()
    problem = PipeMania(board)
    goal_node = depth_first_tree_search(problem)
    goal_node.state.board.print()
    
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.