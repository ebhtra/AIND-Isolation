"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import isolation


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    cx, cy = game.width / 2, game.height / 2
    dx = game.get_player_location(player)[1] - cx
    dy = game.get_player_location(player)[0] - cy
    open_moves = len(game.get_legal_moves(player))
    open_enemy = len(game.get_legal_moves(game.get_opponent(player)))
    return -(dx ** 2) -(dy ** 2) - (open_enemy ** 2) + (open_moves ** 2)
                                 


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='alphabeta', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        
        if not legal_moves:
            return (-1,-1)
        best_move = legal_moves[0]
        best_score = -float('inf')
        depth = 0
        method_map = {'minimax': self.minimax, 'alphabeta': self.alphabeta}
        
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            
            while(self.iterative or depth < self.search_depth):
                
                for move in legal_moves:
                    state = game.forecast_move(move)
                    score, _ = method_map[self.method](state, depth, 
                                         maximizing_player=False)
                    if score > best_score:
                        best_score, best_move = score, move
                depth += 1
            return best_move   
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move
        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        # Set which player the agent is maximizing for, so that
        #    agent can calculate utility of moves correctly
        if maximizing_player:
            boss = game.active_player
        else:
            boss = game.inactive_player
            
        if depth == 0:
            return self.score(game, boss), None
        # if last move is needed instead of None:  game.__last_player_move__[game.inactive_player]
            
        moves = game.get_legal_moves()
        if not moves:
            return game.utility(boss), (-1,-1)
        
        choices = [game.forecast_move(move) for move in moves]
        scores = [self.minimax(choice, depth-1, not maximizing_player)[0]
                          for choice in choices]
        
        if maximizing_player:
            return max(zip(scores, moves), key=lambda x: x[0])
        else:
            return min(zip(scores, moves), key=lambda x: x[0])
            

    def min_val(self, game, maxDepth, target):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        choices = [game.forecast_move(move) for move in game.get_legal_moves()]
        if not choices:
            return float('inf')
        if choices[0].move_count - 2 >= maxDepth:
            return min(self.score(choice, target) for choice in choices)
        return min(self.max_val(choice, maxDepth, target) for choice in choices)
    
    def max_val(self, game, maxDepth, target):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        choices = [game.forecast_move(move) for move in game.get_legal_moves()]
        if not choices:
            return -float('inf')
        if choices[0].move_count - 2 >= maxDepth:
            return max(self.score(choice, target) for choice in choices)
        return max(self.min_val(choice, maxDepth, target) for choice in choices)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        if maximizing_player:
            return self.max_alpha_beta(game, depth, alpha, beta)

        return self.min_alpha_beta(game, depth, alpha, beta)
            

    def max_alpha_beta(self, game, depth, a, b):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        moves = game.get_legal_moves()
        if not moves:
            return -float('inf'), (-1,-1)
        
        if depth == 1:
            best_score, best_move = a, (-1,-1)
            for move in moves:
                board = game.forecast_move(move)
                score = self.score(board, board.inactive_player)
                if score >= b:
                    return score, move
                if score > best_score:
                    best_score, best_move = score, move
                    a = best_score
            return best_score, best_move
        
#        best_score = a
        best_move = (-1,-1)
        for move in moves:
            board = game.forecast_move(move)
            score, _ = self.min_alpha_beta(board, depth-1, a, b)
            if score >= b:
                return score, move
            if score > a:
                a, best_move = score, move
        return a, best_move
    
    def min_alpha_beta(self, game, depth, a, b):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        moves = game.get_legal_moves()
        if not moves:
            return float('inf'), (-1,-1)
        
        if depth == 1:
            best_score, best_move = b, (-1,-1)
            for move in moves:
                board = game.forecast_move(move)
                score = self.score(board, board.active_player)
                if score <= a:
                    return score, move
                if score < best_score:
                    best_score, best_move = score, move
                    b = best_score
            return best_score, best_move
        
#        best_score = b
        best_move = (-1,-1)
        for move in moves:
            board = game.forecast_move(move)
            score, square = self.max_alpha_beta(board, depth-1, a, b)
            if score <= a:
                return score, square
            if score < b:
                b, best_move = score, square
        return b, best_move
    
                
#   