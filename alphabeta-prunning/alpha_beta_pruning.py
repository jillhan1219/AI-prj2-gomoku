import numpy as np
from eval_fn import evaluation_state

class AlphaBetaSearch:
    def __init__(self, hash_table, k = 10):
        self.k = k
        self.hash = hash_table

    def robot(self, chessboard, color, last_drop):

        alpha = float('-inf')
        beta = float('inf')
        max_depth = 2 

        best_move = (-1,-1)
        best_value = float('-inf') if color == 1 else float('inf')
        self.is_max_state = True if color == 1 else False
        #best_value = float('-inf') 

        pieces = len(chessboard[chessboard != 0])
        if pieces == 0:
            best_move =  AlphaBetaSearch.first_move(chessboard)
            new_board = chessboard.copy()
            new_board[best_move[0]][best_move[1]] = color
            self.hash.update_eval_dict(self.hash.calculate_hash(new_board), 1)
            return best_move
        if pieces == 1:
            best_move =  AlphaBetaSearch.second_move(chessboard, last_drop)
            new_board = chessboard.copy()
            new_board[best_move[0]][best_move[1]] = color
            self.hash.update_eval_dict(self.hash.calculate_hash(new_board), 2)
            return best_move

        possible_moves = self.generate_k_moves(chessboard, color, self.is_max_state)

        for move in possible_moves:
            new_board = chessboard.copy()

            new_board[move[0]][move[1]] = color

            score = self.minimax(new_board, max_depth - 1, alpha, beta, False if color==1 else True, -color)

            if (color == 1 and score > best_value) or \
                    (color == -1 and score < best_value):
                best_value = score
                best_move = move

        if best_move[0] == -1 and best_move[1] == -1:
            return possible_moves[0]

        #print(f'best move: {best_move}')
        #print(f'best value: {best_value}')
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player, color):
        # Base case: If the maximum depth is reached or the game is over, return the evaluation score
        if depth == 0 or AlphaBetaSearch.check_win(board):
            score = self.evaluate(board, color)
            return score

        if maximizing_player:
            max_score = float('-inf')

            possible_moves = self.generate_k_moves(board, color, self.is_max_state)

            for move in possible_moves:
                new_board = board.copy()

                new_board[move[0]][move[1]] = color

                score = self.minimax(new_board, depth - 1, alpha, beta, False, -color)

                max_score = max(max_score, score)

                if max_score >= beta:
                    break

                alpha = max(alpha, max_score)

            return max_score
        else:
            min_score = float('inf')

            possible_moves = AlphaBetaSearch.generate_moves(board)

            for move in possible_moves:
                new_board = board.copy()

                new_board[move[0]][move[1]] = color

                score = self.minimax(new_board, depth - 1, alpha, beta, True, -color)

                min_score = min(min_score, score)

                if min_score <= alpha:
                    break

                beta = min(beta, min_score)

            return min_score

    def evaluate(self, board, color):
        board_hash = self.hash.calculate_hash(board)
        if board_hash in self.hash.eval_dict:
            return self.hash.eval_dict[board_hash]

        # Perform evaluation for the current state
        eval_value = evaluation_state(board, color)
        
        self.hash.eval_dict[board_hash] = eval_value
        return eval_value


    def generate_k_moves(self, board, color, is_max_state):
        moves = AlphaBetaSearch.generate_moves(board)

        eval_values = []
        for move in moves:
            new_board = board.copy()
            new_board[move[0]][move[1]] = color
            eval_value = self.evaluate(new_board, color) if is_max_state else -self.evaluate(new_board, color)
            eval_values.append((move, eval_value))

        eval_values.sort(key=lambda x: x[1], reverse=True)
        #print(eval_values)

        top_k_moves = [move for move, _ in eval_values[:self.k]]
        #print(top_k_moves)
        return top_k_moves

    @staticmethod
    def generate_moves(board):
        moves = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == 0:
                    moves.append((i, j))
        return moves

    @staticmethod
    def check_win(board):
        # 检查行
        for row in range(board.shape[0]):
            for col in range(board.shape[1] - 4):
                if np.all(board[row, col:col+5] == -1) or np.all(board[row, col:col+5] == 1):
                    return True

        # 检查列
        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 4):
                if np.all(board[row:row+5, col] == -1) or np.all(board[row:row+5, col] == 1):
                    return True

        # 检查主对角线
        for i in range(board.shape[0] - 4):
            for j in range(board.shape[1] - 4):
                if np.all(np.diag(board[i:i+5, j:j+5]) == -1) or np.all(np.diag(board[i:i+5, j:j+5]) == 1):
                    return True

        # 检查副对角线
        for i in range(board.shape[0] - 4):
            for j in range(4, board.shape[1]):
                if np.all(np.diag(board[i:i+5, j-4:j+1]) == -1) or np.all(np.diag(board[i:i+5, j-4:j+1]) == 1):
                    return True

        return False

    @staticmethod
    def first_move(board):
        x = board.shape[0] // 2
        return tuple(np.random.choice((x - 1, x, x + 1), 2))
    
    @staticmethod
    def second_move(board, last_drop):
        i, j = last_drop
        size = board.shape[0]
        i2 = i <= size // 2 and 1 or -1
        j2 = j <= size // 2 and 1 or -1
        return (i + i2, j + j2)



class ZobristHash:
    def __init__(self, board_size):
        self.board_size = board_size
        self.hash_table = np.random.randint(2**32, size=(board_size, board_size, 2))
        self.eval_dict = {}

    def calculate_hash(self, board):
        hash_value = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = board[i, j]
                if piece != 0:
                    piece_index = 0 if piece == -1 else 1
                    hash_value ^= self.hash_table[i, j, piece_index]
        return hash_value

    def update_eval_dict(self, hash_key, hash_value):
        self.eval_dict[hash_key] = hash_value


'''
class ZobristHash:
    def __init__(self, board_size = 15):
        self.board_size = board_size
        self.MW = np.random.randint(2**32, size=(board_size, board_size))
        self.MB = np.random.randint(2**32, size=(board_size, board_size))
        self.board_hash = 0
        self.eval_dict = {}

    def update_hash(self, i, j, color):
        if color == -1:  # Black player
            self.board_hash ^= self.MB[i][j]
        elif color == 1:  # White player
            self.board_hash ^= self.MW[i][j]

    def calculate_hash(self, i, j, color):
        if color == -1:  # Black player
            return self.board_hash ^ self.MB[i][j]
        elif color == 1:  # White player
            return self.board_hash ^ self.MW[i][j]

    def get_current_hash(self):
        return self.board_hash

    def evaluate(self, depth):
        if self.board_hash in self.eval_dict:
            stored_depth, eval_value = self.eval_dict[self.board_hash]
            if depth >= stored_depth:
                return eval_value

        # Perform evaluation for the current state
        eval_value = self.evaluate_current_state()

        # Store the evaluation value with current depth and board hash
        self.eval_dict[self.board_hash] = (depth, eval_value)

        return eval_value

    def evaluate_current_state(self):
        # Perform evaluation for the current state
        # ...

        # Return the evaluation value
        return eval_value

'''
