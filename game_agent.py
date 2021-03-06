"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

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
    # First check if game is over after this move
    if game.is_winner(player):
        return float("inf")
    # These two game-over checks are in the wrong order in the sample players.
    # Since the game state being scored here is one in which the move has been
    #   forecast already, the next player to move is not the player parameter
    #   here.  So first the opponent must be given the chance to lose, which
    #   translates to checking if game.is_winner(player) first. 
    if game.is_loser(player):
        return float("-inf")

    # What moves does a player have from a given position?
    #  First turn:
    branches = set(game.get_legal_moves(player))
    #  Second turn:
    twigs = set(sum([game.__get_moves__(branch) for branch in branches],[]))

    # What moves does an adversary have from a given position?
    #  First turn:
    enemy_branches = set(game.get_legal_moves(game.get_opponent(player)))
    #  Second turn:
    enemy_twigs = set(sum([game.__get_moves__(branch) for 
                           branch in enemy_branches],[]))
    
    # Score based on how many unstoppable moves player has in ensuing two
    #    turns, vs. same for opponent.
    #
    # The order of turns after the current moves being scored by self are:
    #        1) enemy_branches
    #        2) branches (for self)
    #        3) enemy_twigs
    #        4) twigs (for self) 
    # Any good future moves for either player that can first be taken by
    #    the opponent must be subtracted from player's score.
    return float(  len(twigs - enemy_twigs - enemy_branches)
                 + len(branches - enemy_branches)
                 - len(enemy_branches)
                 - len(enemy_twigs - branches)  )
    
#==============================================================================
#     Unused heuristics follow
#==============================================================================

#    # For the first fraction of the game use the central heuristic
#    cx, cy = game.width / 2, game.height / 2   # The centerpoint of the board
#    # How far off-center is a location? 
#    dx = game.get_player_location(player)[1] + 0.5 - cx
#    dy = game.get_player_location(player)[0] + 0.5 - cy

#    if game.move_count < game.width * game.height / 4.0 :
#        return -(dx ** 2) -(dy ** 2)  # penalize Euclidean distance from center
#    else:
#        # Use a slightly weighted version of "improved" heuristic
#        return 1.3 * len(branches) - len(enemy_branches)


#     # Alternating weights of "improved" heuristic 
#    if game.move_count // 2 % 2 == 0:  # Toggle every 2 plies
#        return 2.0 * len(branches) - len(enemy_branches)                
#    else:
#        return len(branches) - 2.0 * len(enemy_branches)


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
            # game is over
            return (-1,-1)  # game over sentinel
            
        # Get a value to return in case early timeout.
        # This var will store the best move after each deeper iteration.
        last_best = legal_moves[0]
        
        # Map string arguments to the appropriate function calls
        methods = {'minimax': self.minimax, 'alphabeta': self.alphabeta}
        
        # 'unchanged' will be used as a counter to gauge when to
        #      abort a search that seems to have run its course.
        unchanged = 0
        # 'depth' will tell each successive iteration how far to go
        depth = 0
        
        # TODO: Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book)

        try:
            choices = legal_moves[:]
            # Exit loop if iterative is set False AND depth reaches set limit,
            #    OR if best score has stayed the same for several depths
            while((self.iterative or depth < self.search_depth) and 
                                                                unchanged < 7):
                unchanged += 1
                new_scores = []  # This stores new depth scores for each move

                for move in choices:
                    new_board = game.forecast_move(move)
                    f = methods[self.method]  # Translate string to function.
                    # Use the recursive search function to predict the score
                    #     for this branch/move.
                    score, _ = f(new_board, depth, maximizing_player=False)
                    
                    if score == float('inf'):  # This move is a game winner.
                        return move
                    elif score == -float('inf'): # This move is a game loser.
                        # Will be a loser for all successively deeper searches,
                        #    so prune it now.
                        choices.remove(move)
                        if not choices:
                            # All moves will lose, but might as well hope
                            #    opponent doesn't capitalize on that.
                            return last_best[-1]
                    else:
                        new_scores.append((score, move))
                if not new_scores:
                    # Take the best move from previous depth if all new fail
                    return last_best[-1]
                
                new_best = max(new_scores)
                if new_best[0] != last_best[0]:
                    # If the best score for new depth is different from
                    #   best for previous depth, reset unchanged counter.
                    unchanged = 0
                # Update the best move in case loop terminates or times out
                last_best = new_best
                # Get ready to search one move deeper into future games
                depth += 1
            # If score hasn't changed for several depths or if iterative is
            #    set False and search depth has reached max passed in, return
            #    best move when while loop terminates.
            return last_best[-1]
            

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return last_best[-1]
        
        # Return the best move from the last completed search iteration
        return last_best[-1]

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
        # Now call this method recursively, swapping maximizing_player
        scores = [self.minimax(choice, depth-1, not maximizing_player)[0]
                          for choice in choices]
        
        if maximizing_player:
            # Select the move with the highest minimized score
            return max(zip(scores, moves))
        else:
            # Select the move with the lowest maximized score
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
        
        # Set which player the agent is maximizing for, so that
        #    agent can calculate utility of moves correctly
        if maximizing_player:
            boss = game.active_player
            # Init the best score that will be returned eventually
            best_score = alpha
            # The textbook initializes as below instead, but I believe that's
            #    less efficient than above.
            #best_score = -float('inf')
        else:  # (minimizing player is active player)
            boss = game.inactive_player
            best_score = beta
            # Textbook:
            #best_score = float('inf')
            
        if depth == 0:
            # Base case for recursion
            return self.score(game, boss), None
            
        moves = game.get_legal_moves()
        if not moves:
            # game would end here, so return +inf or -inf for score
            return game.utility(boss), (-1,-1)
        
        best_move = (-1,-1)  # Initialize move than can be returned with score
        
        for move in moves:
            
            board = game.forecast_move(move)
            # Call this method recursively, swapping maximizing_player
            score, _ = self.alphabeta(board, depth-1, alpha, beta,
                                   not maximizing_player)
            
            if maximizing_player:
                if score >= beta:
                    # Smallest score opponent could stick this player with on
                    #   this branch is larger than a score that opponent can
                    #   stick this player with on a different branch, so return
                    #   now to not waste time here.
                    return score, move
                elif score > alpha:
                    # Smallest score allowed by opponent is better than what
                    #   this player has scored elsewhere, so make this the
                    #   new benchmark.
                    alpha, best_score, best_move = score, score, move
                # Only need following if initializing best_score as in textbook.
#                elif score > best_score:
#                    best_score, best_move = score, move
            else:  # (minimizing layer)
                if score <= alpha:
                    # Highest score this player gets here is worse
                    #   than what he can get on a different branch, so return
                    #   now in order to prune (not waste time on) the rest of
                    #   this branch, which opponent is minimizing.
                    return score, move
                elif score < beta:
                    # Highest score found by this player is lower than on all
                    #   other branches so far explored, so this score becomes
                    #   the new benchmark for the minimizing player 
                    beta, best_score, best_move = score, score, move
                # Only need following if initializing best_score as in textbook.
#                elif score < best_score:
#                    best_score, best_move = score, move
        return best_score, best_move
        
            