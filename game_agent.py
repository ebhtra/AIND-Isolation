"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
#import random
#import isolation


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
    
#    cx, cy = game.width / 2, game.height / 2
#    dx = game.get_player_location(player)[1] - cx
#    dy = game.get_player_location(player)[0] - cy
    open_moves = len(game.get_legal_moves(player))
    open_enemy = len(game.get_legal_moves(game.get_opponent(player)))
#    if game.move_count < 10:
#        return -(dx ** 2) -(dy ** 2)
#    else:
    return open_moves - 1.5 * open_enemy
                                 


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
                 iterative=True, method='minimax', timeout=10.):
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
            return (-1,-1)  # game over Sentinel
        # List the game boards resulting from each legal move.
        # Note that the active player has been switched in the new board!
        states = [game.forecast_move(move) for move in legal_moves]
        scores = [self.score(state, state.inactive_player) for state in states]
        # Rank the utility of each branch
        ranked_moves = sorted(zip(scores, legal_moves), reverse=True)
        # Keep the initial best move along with its score. This will continue
        #   to update during this method, depth by depth, and be returned
        #   upon exhaustion of search or time.
        last_best = ranked_moves[0]
        # Quick check to see if the best move is a game winner.
        if last_best[0] == float('inf'):
            return last_best[1]
        
        depth = 1  # Already explored one depth
        # Map string arguments to the appropriate function calls
        methods = {'minimax': self.minimax, 'alphabeta': self.alphabeta}
        # 'unchanged' will be used as a counter to gauge quiescence in order
        #      to abort a search that seems to have run its course
        unchanged = 0
        # TODO: Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book)

        try:
            # The loop exits are (not @iterative AND 'depth' > @search_depth)
            #                       OR search result has stopped progressing
            while((self.iterative or depth < self.search_depth) and 
                  unchanged < 6):
                
                unchanged += 1
                
                new_scores = []  # This stores new depth scores for each move
                for rm in ranked_moves:
                    score, _ = methods[self.method](game.forecast_move(rm[1]),
                                         depth, maximizing_player=False)
                    if score == float('inf'):
                        return rm[1]
                    if score == -float('inf'):
                        ranked_moves.remove(rm)
                        if not ranked_moves:
                            return last_best[1]
                    new_scores.append((score, rm[1]))
                
                new_scores = sorted(new_scores, reverse=True)
                last_best = new_scores[0]
                if ranked_moves[0] != new_scores[0]:
                    unchanged = 0
                ranked_moves = new_scores
                depth += 1
            
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            

        except Timeout:
            # Handle any actions required at timeout, if necessary
            
            return last_best[1]
        # Return the best move from the last completed search iteration
#        
        return last_best[1]

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
        # if last move is needed instead of None:  game.get_player_location(game.inactive_player)
            
        moves = game.get_legal_moves()
        if not moves:
            return game.utility(boss), (-1,-1)
        
        choices = [game.forecast_move(move) for move in moves]
        scores = [self.minimax(choice, depth-1, not maximizing_player)[0]
                          for choice in choices]
        
        if maximizing_player:
            return max(zip(scores, moves))
        else:
            return min(zip(scores, moves))
            

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