import numpy as np

class GomokuBoard(object):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    DIR = [[1, 0], [0, 1], [1, 1], [1, -1]]
    # 落子奖励
    common_reward = -1000000  # 非法落子
    win_reward = 1000000  # 胜利
    # 棋盘大小
    SIZE = 15

    def __init__(self):
        self.num = 0
        self.turn = self.BLACK
        self.winner = self.EMPTY
        self.premove = [-1, -1]
        self.board = np.zeros([self.SIZE, self.SIZE])
        self.board[:, :] = self.EMPTY
        self.dir = [[(-1, 0), (1, 0)], [(0, -1), (0, 1)],
                    [(-1, 1), (1, -1)], [(-1, -1), (1, 1)]]

    def printChess(self):
        print("  ", end="")
        for i in range(self.SIZE):
            print("%3d" % (i), end="")
        print("")

        for i in range(self.SIZE):
            print("%2d" % (i), end="")
            for j in range(self.SIZE):
                if self.board[i, j] == 0:
                    print("  *", end="")
                elif self.board[i, j] == 1:
                    print("  o", end="")
                else:
                    print("  -", end="")
            print("")

    def judge_Legal(self, x, y):
        return (x >= 0 and x < self.SIZE) and (y >= 0 and y < self.SIZE)

    def board(self):
        return self.board

    def draw_XY(self, x, y):
        if x == -1 or y == -1:
            return self.board, self.common_reward, True, {}

        # 非法落子
        if (not self.judge_Legal(x, y)):
            return self.board, self.common_reward, False, {}

        self.board[x][y] = self.turn
        self.num += 1
        self.turn = self.turn ^ 1  # 更换落子方
        self.premove = (x, y)
        winner = self.judge_Win()

        if winner == self.EMPTY:  # 没有胜利方
            reward = score((x, y), self.turn, self.board)
            reward += int(score((x, y), self.turn ^ 1, self.board)*0.9)
            return self.board, reward, False, {}
        else:  # 有胜利方
            return self.board, self.win_reward, True, {}

    def judge_Win(self):
        x = 0
        y = 0
        cnt = 0
        color = self.EMPTY
        # 遍历四个方向
        for d in range(4):
            color = self.EMPTY
            cnt = 0
            # 遍历9颗连续棋子
            for k in range(-4, 5):
                x = self.premove[0] + self.DIR[d][0] * k
                y = self.premove[1] + self.DIR[d][1] * k
                if self.judge_Legal(x, y):
                    if self.board[x][y] == self.EMPTY:
                        color = self.EMPTY
                        cnt = 0
                    else:
                        if self.board[x][y] == color:
                            cnt += 1
                        else:
                            color = self.board[x][y]
                            cnt = 1
                else:
                    if k > 0:
                        break
                if cnt == 5:
                    self.winner = color
                    return color
        return self.EMPTY

    def reset(self):
        self.num = 0
        self.turn = self.BLACK
        self.winner = self.EMPTY
        self.board[:, :] = self.EMPTY

score_level = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 1000000]

# pos: (x, y)表示棋盘上的一个位置。这个位置不能是非法的，需要提前判断
# color: 落子的颜色 0和1，0表示黑子，1表示白子
# board: 棋盘 这里的表示方法是一个二维数组（index为0~14），0表示黑子，1表示白子，None表示空
# return: pos位置落子后，本方的奖励值。
#   对于color方来说，一个位置的最终奖励应该为score(pos, color, board) + int(score(pos, 1-color, board)) * 0.9
def score(pos, color, board):
    hori = 1
    verti = 1
    slash = 1
    backslash = 1
    left = pos[0] - 1

    # board加上一行一列
    board = np.insert(board, 0, values=None, axis=0)
    board = np.insert(board, 0, values=None, axis=1)

    while left > 0 and board[left][pos[1]] == color:
        left -= 1
        if hori == 4:
            hori += 1
            break
        if left > 0 and \
                (board[left][pos[1]] == color or
                 board[left][pos[1]] is None):
            hori += 1

    right = pos[0] + 1
    while right < 16 and board[right][pos[1]] == color:
        right += 1
        if hori == 4:
            hori += 1
            break
        if right < 16 and \
                (board[right][pos[1]] == color or
                 board[right][pos[1]] is None):
            hori += 1

    hori = score_level[hori]

    up = pos[1] - 1
    while up > 0 and board[pos[0]][up] == color:
        up -= 1
        if verti == 4:
            verti += 1
            break
        if up > 0 and \
                (board[pos[0]][up] == color or
                 board[pos[0]][up] is None):
            verti += 1

    down = pos[1] + 1
    while down < 16 and board[pos[0]][down] == color:
        down += 1
        if verti == 4:
            verti += 1
            break
        if down < 16 and \
                (board[pos[0]][down] == color or
                 board[pos[0]][down] is None):
            verti += 1

    verti = score_level[verti]

    left = pos[0] - 1
    up = pos[1] - 1
    while left > 0 and up > 0 and board[left][up] == color:
        left -= 1
        up -= 1
        if backslash == 4:
            backslash += 1
            break
        if left > 0 and up > 0 and \
                (board[left][up] == color or
                 board[left][up] is None):
            backslash += 1

    right = pos[0] + 1
    down = pos[1] + 1
    while right < 16 and down < 16 and board[right][down] == color:
        right += 1
        down += 1
        if backslash == 4:
            backslash += 1
            break
        if right < 16 and down < 16 and \
                (board[right][down] == color or
                 board[right][down] is None):
            backslash += 1
    backslash = score_level[backslash]

    right = pos[0] + 1
    up = pos[1] - 1
    while right < 16 and up > 0 and board[right][up] == color:
        right += 1
        up -= 1
        if slash == 4:
            slash += 1
            break
        if right < 16 and up > 0 and (board[right][up] == color or
                                      board[right][up] is None):
            slash += 1

    left = pos[0] - 1
    down = pos[1] + 1
    while left > 0 and down < 16 and board[left][down] == color:
        left -= 1
        down += 1
        if slash == 4:
            slash += 1
            break
        if left > 0 and down < 16 and (board[left][down] == color or
                                       board[left][down] is None):
            slash += 1

    slash = score_level[slash]

    return hori + verti + slash + backslash
