import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.tictactoe import TicTacToeGameState


def test_stop_early_if_only_one_move():
    # Check that it stops early, if there is only one possible move:
    gamestate = np.array((( 1, -1, -1),
                          (-1,  1,  1),
                          ( 1,  0, -1)))
    next_to_move = 1
    initial_game_state = TicTacToeGameState(gamestate, next_to_move)
    game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
    mcts = MonteCarloTreeSearch(game)
    best_node = mcts.best_action(100)

    assert np.where(abs(best_node.state.board - gamestate) > 0) == (2, 1)
    assert mcts.root.n <= 1


def test_stop_early_if_only_two_fields():
    # Check that it stops early, if there are only two possible moves:
    gamestate = np.array((( 1, -1, -1),
                          (-1,  1,  1),
                          ( 1,  0,  0)))
    next_to_move = -1
    initial_game_state = TicTacToeGameState(gamestate, next_to_move)
    game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
    mcts = MonteCarloTreeSearch(game)
    best_node = mcts.best_action(100)  # worst case 9*8*7*6*5*4*3*2*1 = 362880 children

    assert np.where(abs(best_node.state.board - gamestate) > 0) == (2, 2)
    assert mcts.root.n <= 4


def test_stop_early_if_only_three_fields():
    # Check that it stops early, if there are only three possible moves:
    gamestate = np.array((( 1,  0, -1),
                          (-1,  1,  0),
                          ( 1,  0, -1)))
    next_to_move = 1
    initial_game_state = TicTacToeGameState(gamestate, next_to_move)
    game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
    mcts = MonteCarloTreeSearch(game)
    best_node = mcts.best_action(100)  # worst case 9*8*7*6*5*4*3*2*1 = 362880 children

    assert np.where(abs(best_node.state.board - gamestate) > 0) == (1, 2)
    assert mcts.root.n <= 13  # 4 + 5 + 4


def test_stop_if_winning_move():
    # Check that it stops, if there is a direct win possible:
    gamestate = np.array((( 1,  0,  0),
                          ( 0,  1,  0),
                          ( 0,  0,  0)))
    next_to_move = 1
    initial_game_state = TicTacToeGameState(gamestate, next_to_move)
    game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
    mcts = MonteCarloTreeSearch(game)
    best_node = mcts.best_action(100)  # worst case 2*7*6*5*4*3*2*1 = 362880 children

    assert np.where(abs(best_node.state.board - gamestate) > 0) == (2, 2)
    assert mcts.root.n < 10


def test_if_wins():
    # Check that it finds a secure winning strategy:
    gamestate = np.array((( 0,  0,  0),
                          ( 0,  1, -1),
                          ( 0,  0,  0)))
    next_to_move = 1
    initial_game_state = TicTacToeGameState(gamestate, next_to_move)
    game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
    mcts = MonteCarloTreeSearch(game)
    best_node = mcts.best_action(1000)  # worst case 9*8*7*6*5*4*3*2*1 = 362880 children

    best_move = np.where(abs(best_node.state.board.flatten() - gamestate.flatten()) > 0)[0]
    print(best_move)
    assert (best_move in (0,2,6,8))
    assert mcts.root.n < 1000


def test_if_stop_early():
    # Check that it does not terminate too early:
    gamestate = np.array((( 0,  0,  0),
                          ( 0,  0,  0),
                          ( 0,  0,  0)))
    next_to_move = 1
    initial_game_state = TicTacToeGameState(gamestate, next_to_move)
    game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
    mcts = MonteCarloTreeSearch(game)
    best_node = mcts.best_action(2000)  # worst case 9*8*7*6*5*4*3*2*1 = 362880 children

    best_move = np.where(abs(best_node.state.board - gamestate) > 0)
    print(best_move)
    assert ((best_move[0] == 1) & (best_move[1] == 1))
    assert mcts.root.n >= 1468
    assert not mcts.root.has_child_with_untried_actions


def test_does_not_stop_too_early():
    gamestate = np.array(((0, 0, -1),
                          (0, 1, 0),
                          (0, 0, 0)))
    next_to_move = 1
    initial_game_state = TicTacToeGameState(gamestate, next_to_move)
    game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
    mcts = MonteCarloTreeSearch(game)
    best_node = mcts.best_action(1000)
    assert not mcts.root.has_child_with_untried_actions

