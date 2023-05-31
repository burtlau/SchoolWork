# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDis = float('inf')
        for ghost in newGhostStates:
            if ghostDis > util.manhattanDistance(newPos, ghost.getPosition()):
                ghostDis = util.manhattanDistance(newPos, ghost.getPosition())
        if ghostDis < 2:
            return -float('inf')

        minFood = float('inf')
        for foodPosition in currentGameState.getFood().asList():
            distance = util.manhattanDistance(newPos, foodPosition)
            if distance < minFood:
                minFood = util.manhattanDistance(newPos, foodPosition)
        if minFood == 0:
            return float('inf')
        return ghostDis / minFood + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, agent, state, depth):
        # terminal state
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP
        actions = state.getLegalActions(agent)

        # agent is pacman
        if agent == 0:
            scores = []
            for action in actions:
                new_state = state.generateSuccessor(agent, action)
                scores.append(self.minimax(1, new_state, depth - 1)[0])
            index_array = []
            max_score = max(scores)
            for i in range((len(scores))):
                if scores[i] == max_score:
                    index_array.append(i)
            return max_score, actions[random.choice(index_array)]

        # agent is ghost
        else:
            scores = []
            for action in actions:
                if agent == state.getNumAgents() - 1:
                    new_agent = 0
                    new_depth = depth - 1
                else:
                    new_agent = agent + 1
                    new_depth = depth
                new_state = state.generateSuccessor(agent, action)
                scores.append(self.minimax(new_agent, new_state, new_depth)[0])

            index_array = []
            min_score = min(scores)
            for i in range((len(scores))):
                if scores[i] == min_score:
                    index_array.append(i)
            return min_score, actions[random.choice(index_array)]

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.minimax(0, gameState, self.depth*2)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self, agent, state, alpha, beta, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP

        bestMove = None
        if agent == 0:
            v = float('-inf')
            next_agent = 1
        else:
            v = float('inf')
            if agent == state.getNumAgents() - 1:
                next_agent = 0
            else:
                next_agent = agent + 1
        actions = state.getLegalActions(agent)
        if agent == 0 or agent == state.getNumAgents() - 1:
            new_depth = depth - 1
        else:
            new_depth = depth

        for action in actions:
            next_state = state.generateSuccessor(agent, action)
            next_value, next_action = self.alphaBeta(next_agent, next_state, alpha, beta, new_depth)
            if agent == 0:
                if v < next_value:
                    v, bestMove = next_value, action
                if v >= beta:
                    return v, bestMove
                alpha = max(alpha, v)
            if agent >= 1:
                if v > next_value:
                    v, bestMove = next_value, action
                if v <= alpha:
                    return v, bestMove
                beta = min(beta, v)
        return v, bestMove


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(0, gameState, float('-inf'), float('inf'), self.depth * 2)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action = self.Expectimax(0, gameState, self.depth)[1]
        return action
        util.raiseNotDefined()

    def Expectimax(self, agent, state, depth):
        final_action = Directions.STOP
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), Directions.STOP

        # agent is Pacman
        if agent == 0:
            max_value = float('-inf')
            new_agent = 1

            for action in state.getLegalActions(agent):
                new_state = state.generateSuccessor(agent, action)
                value = self.Expectimax(new_agent, new_state, depth)[0]
                if value > max_value:
                    max_value = value
                    final_action = action

        # agent is Ghost
        else:
            if agent == state.getNumAgents() - 1:
                new_depth = depth - 1
                new_agent = 0
                max_value = 0
            else:
                new_depth = depth
                max_value = 0
                new_agent = agent + 1
            for action in state.getLegalActions(agent):
                new_state = state.generateSuccessor(agent, action)
                chance = 1 / len(state.getLegalActions(agent))
                expectation = chance * self.Expectimax(new_agent, new_state, new_depth)[0]
                max_value += expectation
                final_action = action

        return max_value, final_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    if ghost is very close to pacman( < 2) : evaluate the lowest to -infinity
    # the minimum distance to food : the less the better
    # the remaining number of food : the less the better
    # the remaining number of capsules : the less the better
    # the score : the more the better
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostPosistions = currentGameState.getGhostPositions()
    newCapsules = currentGameState.getCapsules()
    foodNum = currentGameState.getNumFood()

    # if ghost get closer than 2, let the evaluation be extremely low
    for ghost in newGhostPosistions:
        ghostDis = util.manhattanDistance(newPos, ghost)
        if ghostDis < 2:
            return -float('inf')

    minDis = float('inf')
    score = currentGameState.getScore()
    for food in newFood.asList():
        foodDis = util.manhattanDistance(newPos, food)
        if foodDis < minDis:
            minDis = foodDis

    if minDis == float('inf'):
        minDis = 0

    # the minimum distance to food : the less the better
    # the remain of food : the less the better
    # the remain of capsules : the less the better
    # the score : the more the better
    return -minDis - 300 * foodNum - 2 * len(newCapsules) + 2 * score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
