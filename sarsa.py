import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.low = state_low
        self.high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_actions = num_actions
        self.grid = []
        self.grid_size = 0
        self.d = 0
        #iterate over tilings
        for a in range(self.num_actions):
            mini_grid = []
            for tilling_index in range(self.num_tilings):
                tile = []
                num_tiles_cat = []
                for dimension_index in range(len(self.low)):
                    high = self.high[dimension_index]
                    low = self.low[dimension_index]
                    tile_width = self.tile_width[dimension_index]

                
                    num_tiles = int(np.ceil((high - low) / tile_width) + 1)
                    start = low - tilling_index/self.num_tilings * tile_width

                    num_tiles_cat.append(num_tiles)
                    
                    dim = [start]
                    for _ in range(num_tiles -1):
                        dim.append(dim[-1] + tile_width)

                    tile.append(dim)
                mini_grid.append(tile)
            self.grid.append(mini_grid)

        self.grid = np.array(self.grid)

    
    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        for tilling_index in range(self.num_tilings):
            num_tiles_cat = []
            for dimension_index in range(len(self.low)):
                high = self.high[dimension_index]
                low = self.low[dimension_index]
                tile_width = self.tile_width[dimension_index]

                num_tiles = int(np.ceil((high - low) / tile_width) + 1)
                start = low - tilling_index/self.num_tilings * tile_width

                num_tiles_cat.append(num_tiles)
                
        d = [self.num_actions, self.num_tilings] + num_tiles_cat
        self.grid_size = d
        
        self.d = np.prod(d)
        return np.prod(d)
        


    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        if done:
            return np.zeros(self.d)
        else:
            select_grid = self.grid[a]
            cells = []
            for i in range(self.num_tilings):
                tile_idx = []
                for j in range(len(self.low)): #naviagate through grid to the state and weight values for each tiling
                    state = s[j]
                    bins = select_grid[i, j]
                    k = np.digitize(state ,bins) #informs which indices the state falls into
                    tile_idx.append(k)
                
                cell = tuple([a ,i] + tile_idx)
                cells.append(cell) # pointer to the specific weights and values being used
        

            feats = np.zeros(tuple(self.grid_size))
            for i in range(len(cells)):
                feats[cells[i]] = 1
        

        return feats.flatten()


def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """
    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))


    for i in range(num_episode):
        env.reset()

        state = env.state 
        action = epsilon_greedy_policy(state, False, w, .01)

        x = X(state, False, action)
    
        z = np.zeros_like(x)
        q_old = 0

        done = False
        while not done:
            next_state, reward, done, info = env.step(action)
    
            next_action = epsilon_greedy_policy(next_state, done, w)
            
            next_x = X(next_state, done, next_action)

            Q = np.dot(w, x)
            next_Q = np.dot(w, next_x)

            delta = reward + gamma* next_Q - Q

            z = (lam * gamma * z) +  (1 - (alpha * lam * gamma * np.dot(z, x))) * x

            w += (alpha * (delta + Q - next_Q) * z) - (alpha * (Q - next_Q) * x)

            q_old = Q
            x = next_x
            action = next_action

            if done:
                break
        

    print(w)   
    return w