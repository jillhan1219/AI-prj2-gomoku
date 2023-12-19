from alpha_beta_pruning import ZobristHash, AlphaBetaSearch
zobrist_hash = ZobristHash(15)

def robot(chessboard, robot_color, last_drop):
    # chessboard    a 15*15 ndarray, 0 is empty, -1 is black, 1 is white
    # robot_color   a interger, -1 is black, 1 is white
    # last_drop     a tuple (r,c), r is row of chessboard, c is column of chessboard

    # RETURN:       a tuple (r,c), which is location of robot to drop piece in this turn

    alpha_bata_search = AlphaBetaSearch(zobrist_hash, 10)
    best_drop = alpha_bata_search.robot(chessboard, robot_color, last_drop)

    return best_drop
    #return (r, c)
