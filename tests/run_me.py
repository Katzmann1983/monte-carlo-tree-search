import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.tictactoe import TicTacToeGameState
import time

state = np.array(((0, -1, 0),
                  (0, 1, 0),
                  (0, 1, -1)))

next_to_move = 1
initial_game_state = TicTacToeGameState(state, next_to_move)
game = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_game_state)
mcts = MonteCarloTreeSearch(game)
print(f"mcts: {mcts}")
start = time.time()
best_node = mcts.best_action(1000)  # worst case 5*4*3*2*1 = 362880 children
end = time.time()
print(end - start)
print(f"mcts: {mcts}")
print(f"best_node: \n{best_node}")
print(f"best_node.state: \n{best_node.state}")
print(f"best_node.children: \n{best_node.children}")


# Make a tree of the game state
print ("Trees")
from treelib import Tree, Node
# Draw all the cildren
tree = Tree()
rootid = 1
root = mcts.root  # best_node


def add_to_tree(node, parentid):
    for j, c in enumerate(node.children):
        id = parentid * 10 + j
        tree.create_node(str(c), id, parent=parentid)
        add_to_tree(c, id)


tree.create_node(str(root), rootid)
add_to_tree(root, rootid)

tree.show()
