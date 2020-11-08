import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, drconda install -c conda-forge gymaw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, natural=False, double=False):
        self.double = double
        self.action_space = spaces.Discrete(3) if double else spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        elif action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        elif action == 2:
            self.player.append(draw_card(self.np_random))
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
            reward *= 2

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()


class Deck:
    def __init__(self):
        self.cards = None
        self.reset()

    def reset(self):
        self.cards = [i for i in range(1, 11)]*4
        self.cards += (4*3)*[10]

    def draw(self):
        reseted = False
        n = np.random.randint(0, len(self.cards))
        card = self.cards[n]
        del self.cards[n]
        if len(self.cards) == 0:
            self.reset()
            reseted = True
        return card, reseted


class MyBlackjackEnv:
    def __init__(self, natural=False):
        self.deck = Deck()
        self.natural = natural
        self.player = []
        self.dealer = []
        self.cards = []

    def step(self, action):
        reseted = False
        if action == 1:  # hit: add a card to players hand and return
            card, reseted_ = self.deck.draw()
            self.cards.append(card)
            if reseted_:
                self.cards = []
                reseted = True
            self.player.append(card)
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        elif action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                card, reseted_ = self.deck.draw()
                self.cards.append(card)
                self.dealer.append(card)
                if reseted_:
                    self.cards = []
                    reseted = True

            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        elif action == 2:
            card, reseted_ = self.deck.draw()
            self.cards.append(card)
            if reseted_:
                self.cards = []
                reseted = True
            self.player.append(card)
            done = True
            while sum_hand(self.dealer) < 17:
                card, reseted_ = self.deck.draw()
                self.cards.append(card)
                if reseted_:
                    self.cards = []
                    reseted = True
                self.dealer.append(card)
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
            reward *= 2

        return self._get_obs(), reward, done, reseted

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        reseted = False
        self.player = []
        self.dealer = []

        for i in range(2):
            card, reseted_ = self.deck.draw()
            self.cards.append(card)
            if reseted_:
                self.cards = []
                reseted = True
            self.dealer.append(card)
        for i in range(2):
            card, reseted_ = self.deck.draw()
            self.cards.append(card)
            if reseted_:
                self.cards = []
                reseted = True
            self.player.append(card)
        return self._get_obs(), 0, False, reseted

    def cards_to_index(self):
        index = [0]*10
        for card in self.cards:
            index[card-1] += 1
        return index
















