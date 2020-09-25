"""2048 game, based on http://gabrielecirulli.github.io/2048/.

"""

import enum
import numpy as np
import random
import os
import datetime
import math
import time
 # TODO(): Add other imports

WINNING_SCORE = 2048
ACTIONS = {"L": 0, "R": 1, "U": 2, "D": 3}


class SpawnType(enum.Enum):
  RANDOM = 1
  ALWAYS_2 = 2
  ALWAYS_4 = 3


class SpawnLocation(enum.Enum):
  RANDOM = 1
  FIRST_AVAILABLE = 2


def add_new_tile(board, spawn_type=SpawnType.RANDOM,
                 spawn_location=SpawnLocation.RANDOM):

  """A new tile is spawned uniformly among empty squares.

  By default, a new tile has value 4 w.p. 0.1 and 2 w.p. 0.9. No action is
  taken if there are zero empty squares on board.

  Args:
    board: numpy array containing board position.
    spawn_type: the value of the new tile (random by default).
    spawn_location: the location of the new tile (random by default).

  Raises:
    ValueError: if spawn_type or spawn_location is unknown.
  """

   # TODO(): Fill in
   #TODO buff as 16 long buffer
  zeros = np.argwhere(board == 0)
  num_zeros = np.shape(zeros)[0]
  if num_zeros == 0:
    return

  if(spawn_type == SpawnType.RANDOM):
    tile_val = np.random.choice([4,2], p=[0.1,0.9])
  elif(spawn_type == SpawnType.ALWAYS_2):
    tile_val = 2
  elif(spawn_type == SpawnType.ALWAYS_4):
    tile_val = 4
  else:
    raise ValueError("Spawn Type Unknown")

  if(spawn_location == SpawnLocation.FIRST_AVAILABLE):
    new_loc = zeros[0]
  elif(spawn_location == SpawnLocation.RANDOM):
    empty_tile = random.randint(0,num_zeros)-1 #Randint is inclusive for lower and upper bounds
    new_loc = zeros[empty_tile]
  else:
    raise ValueError("Spawn Location Unkown")

  board[new_loc[0], new_loc[1]] = tile_val


def collapse(row):
  """Slides tiles over empty squares in a given row.

  Collapses from high indices to low.

  Args:
    row: numpy array containing a single row. Will be modified in-place.
  """

  # TODO(): fill in
  zeros = np.argwhere(row == 0)
  num_zeros = np.shape(zeros)[0]
  non_zeros = np.where(row != 0)[0]
  temp = []
  if num_zeros > 0:
    for each in non_zeros:
        temp.append(row[each])
    temp.extend([0 for i in range(num_zeros)])
    row[:] = np.array(temp)

def row_move(row):
  """Slides tiles over empty squares and merges equal adjacent tiles.

  Slides from high indices to low.

  Args:
    row: numpy array containing a single row. Will be modified in-place.

  Returns:
    The score corresponding to the total value of combined tiles.
  """

  # TODO(): fill in
  collapse(row)
  i = 0
  score = 0
  new_row = []
  while i < np.shape(row)[0]:
      if i != np.shape(row)[0] - 1 and row[i] == row[i+1] and row[i] != 0:
          new_row.append(row[i]*2)
          score += row[i]*2
          i += 2
      else:
          new_row.append(row[i])
          i += 1
  new_row.extend([0 for i in range((np.shape(row)[0])-len(new_row))])
  row[:] = np.array(new_row)

  return score


def move(board, action):
  """Performs a move on the board.

  Args:
  board: numpy array containing board position. Will be modified in-place.
  action: integer from 1 to 4, which are the values in the dictionary
  ACTIONS

  Returns:
  The score corresponding to the total value of combined tiles.

  Raises:
  ValueError: Passed action is invalid.
   """

  num_rows = np.shape(board)[0]
  num_columns = np.shape(board)[1]
  t_board = board.T
  score = 0

  if action == ACTIONS["L"]:
    for i in range(0, num_rows):
      score += row_move(board[i])

  elif action == ACTIONS["R"]:
    for i in range(0, num_rows):
        row  = board[i]
        reverse = row[::-1]
        score += row_move(reverse)

  elif action == ACTIONS["U"]:
    for i in range(0, num_rows):
        score += row_move(t_board[i])
        board.T

  elif action == ACTIONS["D"]:
    for i in range(0, num_rows):
        row = t_board[i]
        reverse = row[::-1]
        score += row_move(reverse)
        board.T
  else:
    raise ValueError("Passed action is invalid")

  return score

def valid_move(current_board, next_board):
  """A "valid" move is one where at least one tile will move/merge.

  Args:
    current_board: numpy array.
    next_board: numpy array.

  Returns:
    Whether the two board positions are different.
  """

  # TODO(): fill in
  return (current_board == next_board).all()


def can_move(current_board, next_board=None):
  """Returns true if there exists a valid move in any direction.

  Args:
    current_board: numpy array.
    next_board: optional, numpy array used for storing the board position
        upon trying moves.

  Returns:
    Whether there exists a valid move.
  """

  # TODO(): fill in
  valid = False
  for i in range(4):
      next_board = current_board.copy()
      move(next_board, i)
      if  not valid_move(current_board, next_board):
          valid = True
          break

  return valid

def valid_moves(board): #Returns an array containing valid moves for the current board state
  valid = []
  for action in range(4):
      next_board = board.copy()
      move(next_board, action)
      if not valid_move(board, next_board):
          valid.append(action)

  return valid

def max_zeros(states):
    final_move = 0
    final_state = 0
    zeros = 0
    for move, state in states:
        state_zeros = np.count_nonzero(state==0)
        if state_zeros > zeros:
            zeros = state_zeros
            final_move = move
            final_state = state
    if zeros > 0:
        return final_move, final_state
    else:
        final_move, final_state = random.choice(states)
        return final_move, final_state


class MonteCarlo:

    def __init__(self, board, time=6, max_moves=400, C=1.4):
        self._board = board
        self.calculation_time = time
        self.max_moves = max_moves
        self.c = C
        self.states = []
        self.scores = {}
        self.plays = {}

    def update(self, state):
        self.states.append(state)

    def get_play(self):
        state = self.states[-1]
        val_moves = valid_moves(self._board._board)

        if len(val_moves) == 1:
            return val_moves[0]

        games = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            self.run_simulation()
            games += 1

        moves_states = []
        for move in val_moves:
           self._board.__init__(initial_state=state)
           self._board.reset()
           moves_states.append((move, self._board.step(move)[0]))

        move_mean_score = [self.scores.get(hash((tuple(map(tuple, state)))), 0) /
            self.plays.get(hash((tuple(map(tuple, state)))), 1) for move, state in moves_states]

        move = val_moves[np.argmax(move_mean_score)]

        return move

    def run_simulation(self):
        visited_states = set()
        states_copy = self.states[:]
        state = states_copy[-1]
        self._board.__init__(initial_state=state)
        self._board.reset()

        expand = True
        for t in range(self.max_moves):
            val_moves = valid_moves(self._board._board)

            moves_states = []
            for move in val_moves:
                self._board.__init__(initial_state=state)
                self._board.reset()
                moves_states.append((move, self._board.step(move)[0]))

            self._board.__init__(initial_state=state)
            self._board.reset()

            ucb = True
            for move, state in moves_states:
              if hash((tuple(map(tuple, state)))) in self.plays.keys():
                continue
              else:
                ucb = False
                break

            if ucb == True:
              log_total = math.log(sum(self.plays[hash((tuple(map(tuple, state))))] for move, state in moves_states))
              value, move, state = max(
                  ((self.scores[hash((tuple(map(tuple, state))))] / self.plays[hash((tuple(map(tuple, state))))])
                   + self.c * math.sqrt(log_total / self.plays[hash((tuple(map(tuple, state))))]), move, state)
                  for move, state in moves_states)
            else:
                move, state = max_zeros(moves_states)

            state, reward = self._board.step(move)
            states_copy.append(hash((tuple(map(tuple, state)))))

            if expand and hash((tuple(map(tuple, state)))) not in self.plays:
                expand = False
                self.plays[hash((tuple(map(tuple, state))))] = 0
                self.scores[hash((tuple(map(tuple, state))))] = 0

            visited_states.add((hash((tuple(map(tuple, state)))), reward))

            finished = self._board._reset_next_step
            if finished:
                break

        for state, reward in visited_states:
            if state not in self.plays:
                continue
            self.plays[state] += 1
            self.scores[state] += reward

class TwentyFortyEight:
  """2048 game class.

  To use this class with a planning algorithm such as
  [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search), set the
  `initial_state` constructor argument. This will create a 2048 game that always
  starts at the specified state.
  """

  def __init__(self, shape=None, reward_type="score", episodic=False,
               deterministic=False, initial_state=None):
    """Initializes a new `TwentyFortyEight` game.

    Args:
      shape: Optional custom 2D shape of the game board. Defaults to `(4, 4)`.
        If `initial_state` is supplied, the value of `shape` will be overridden
        to match.
      reward_type: One of "score" (default), "max", or "win_loss". "score" gives
        the difference in the regular 2048 score, which is the sum of values of
        all tiles merged in a step. "max" gives the value when a new max tile is
        reached, and is 0 otherwise. "win_loss" gives a 1 if there is a 2048
        tile on the board, -1 if there are no valid moves, and 0 otherwise.
      episodic: Default `False`. Whether to terminate the episode once the 2048
        tile is reached. If `reward_type` is "win_loss" this will be set to
        `True`.
      deterministic: Default `False`, whether to play a deterministic variant of
        the game where new tiles are always placed in the first available slot.
      initial_state: Optional array-like board state. If supplied, the first
        observation of every episode will contain this board state. Must have
        dimensions matching `shape`.

    Raises:
      ValueError: If invalid string passed for `reward`.
      ValueError: If `shape` is not length 2.
      ValueError: If `initial_state` is a terminal state.
    """
    if initial_state is None:
      self._initial_state = None
      shape = tuple(shape if shape is not None else (4, 4))
    else:
      self._initial_state = np.array(initial_state, dtype=np.int)
      shape = self._initial_state.shape

    if len(shape) != 2:
      raise ValueError("Expected a shape of length 2. Got {}.".format(shape))

    if self._initial_state is not None and not can_move(self._initial_state):
      raise ValueError("initial_state must not be a terminal state.")

    self._board = np.zeros(shape, dtype=np.int)

    if reward_type not in ("score", "max", "win_loss"):
      raise ValueError("Invalid reward type {!r}".format(reward_type))

    if reward_type == "win_loss":
      episodic = True

    self._episodic = episodic
    self._deterministic = deterministic
    self._reward_type = reward_type
    self._reset_next_step = True

  def reset(self):
    """Starts a new episode and returns the first `TimeStep`."""
    self._reset_next_step = False
    if self._initial_state is not None:
      self._board[:] = self._initial_state
    else:
      self._board.fill(0)
      self._add_new_tile(self._board)
      self._add_new_tile(self._board)

    return self._board, 0

  def step(self, action):
    """Updates the environment according to action and returns a `TimeStep`."""
    # Handle any reasonable input format.
    action = np.array(action, dtype=np.int).item()

    if self._reset_next_step:
      return self.reset()

    old_board = self._board.copy()

    move_score = move(self._board, action)

    if not (self._board == old_board).all():
      self._add_new_tile(self._board)

    reached_limit = self._episodic and self._board.max() == WINNING_SCORE

    stuck = not can_move(self._board)

    if self._reward_type == "score":
      reward = float(move_score)
    if self._reward_type == "max":
      reward = float(self._board.max() - old_board.max())
    elif self._reward_type == "win_loss":
      if reached_limit:
        reward = 1.0
      elif stuck:
        reward = -1.0
      else:
        reward = 0.0

    if reached_limit or stuck:
      self._reset_next_step = True

    return self._board, reward

  def _add_new_tile(self, board):
    if self._deterministic:
      add_new_tile(board,
                   spawn_type=SpawnType.ALWAYS_2,
                   spawn_location=SpawnLocation.FIRST_AVAILABLE)
    else:
      add_new_tile(board)

  def __str__(self):
    return str(self._board)


def main():
  game = TwentyFortyEight()
  monte = MonteCarlo(game)
  score = 0
  start_time = time.time()

  game.reset()
  monte.update(game._board)

  while can_move(game._board) and time.time() - start_time < 3600:
      move = monte.get_play()
      game.reset()
      state, reward = game.step(move)
      monte.update(state)
      score += reward
      print(state)
      print("Score: {}".format(score))
  print("Final Board State: ")
  print(game._board)
  print("Final Score: {}".format(score))
  sum_of_tiles = np.sum(game._board)
  print("Sum of tiles: {}".format(sum_of_tiles))
  print("Program ran for {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
  main()
