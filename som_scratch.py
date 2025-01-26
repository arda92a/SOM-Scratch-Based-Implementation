import numpy as np
from pylab import bone, pcolor, colorbar, plot, show

class SOM:
    def __init__(self, grid_size, input_dim, learning_rate=0.1, max_iterations=100, radius=1.0, min_learning_rate=0.01, min_radius=0.1,distance_metric="euclidean"):

        """
        Initializes the initial parameters of the class.
        - grid_size: SOM grid size (e.g. (10, 10)).
        - input_dim: Size of input vectors.
        - learning_rate: Initial learning rate.
        - radius: Initial neighborhood radius (optional).
        - max_iterations: Total number of iterations in the training process.
        - distance_metric: Distance metric to use ('euclidean', 'manhattan', etc.).
        """
        if grid_size[0] <= 0 or grid_size[1] <= 0:
            raise ValueError("Grid size dimensions must be positive integers.")
        if input_dim <= 0:
            raise ValueError("Input dimension must be a positive integer.")
        if max_iterations <= 0:
            raise ValueError("Max iterations must be a positive integer.")
        
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']

        if distance_metric not in valid_metrics:
            raise ValueError("Unsupported distance metric: {distance_metric}. Supported metrics: {valid_metrics}")
        
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.initial_radius = radius if radius else max(grid_size)/2
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric
        self.weights = None
        self.neuron_map = {}
        self.min_learning_rate = min_learning_rate  
        self.min_radius = min_radius  




    def random_weights_init(self,seed=None):
        """
        Initializes the weights of neurons in the SOM grid randomly.
        If a seed is provided, the random initialization will be deterministic.
        """

        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.rand(self.grid_size[0],self.grid_size[1],self.input_dim)

    def activation_distance(self, input_vector):
        """
        Calculate the distance between the input vector and the weights of the neurons.
        - input_vector: Input data.
        - weights: Weights of the neurons.
        - return: Distances calculated based on the distance criterion.
        """
        input_vector = input_vector.reshape(1,1,self.input_dim)
        diff = self.weights - input_vector

        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum(diff ** 2, axis=2)) 

        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(diff), axis=2)
        
        elif self.distance_metric == "cosine":
            dot_product = np.sum(input_vector * self.weights, axis=2)
            norm_input_vector = np.linalg.norm(input_vector)
            norm_weights = np.linalg.norm(self.weights, axis=2)
            distance = 1 - (dot_product) / ( norm_input_vector * norm_weights + 1e-10)
            return distance

        elif self.distance_metric == "chebyshev":
            return np.max(np.abs(diff), axis=2) 
        else:
            valid_metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}. Supported metrics: {valid_metrics}") 

    def winner(self, input_vector):
        """
        Finds the neuron (BMU) closest to the input vector.
        - input_vector: Input data.
        - return: Coordinates of BMU on the grid (e.g. (5, 3)).
        """

        if self.weights is None:
            raise ValueError("Weights must be initialized!")
        
        distances = self.activation_distance(input_vector)

        if distances.shape != self.grid_size:
            raise ValueError(f"Distances shape {distances.shape} does not match grid size {self.grid_size}")

        i,j = np.unravel_index(np.argmin(distances),self.grid_size)

        return (i,j)

    def win_map(self, data):
        """
        After the training process, assigns each input vector to the winning neuron.
        - data: Input data.
        - return: A dictionary that returns the winning input vectors for each neuron.
        """

        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("Data should be a list or a numpy array of input vectors.")
        
        winner_map = {}

        for input_vector in data:
            i,j = self.winner(input_vector)

            if (i,j) not in winner_map:
                winner_map[(i,j)] = []
            winner_map[(i, j)].append(input_vector)

        return winner_map
             
    def neighborhood_function(self, winner_i, winner_j, i, j, epoch):
        """
        Computes the neighborhood value for a given neuron (i, j) based on its distance 
        to the winning neuron (winner_i, winner_j) and the current epoch.
        - winner_i, winner_j: Coordinates of the winning neuron (BMU).
        - i, j: Coordinates of the current neuron.
        - epoch: The current epoch number.
        - return: The neighborhood value.
        """
        radius = self.initial_radius * (1 - epoch / self.max_iterations)
        radius = max(radius, self.min_radius)
        
        distance = np.sqrt((winner_i - i) ** 2 + (winner_j - j) ** 2)

        if radius == 0:  
            return 0

        neighborhood_value = np.exp(-distance**2 / (2 * radius**2))

        return neighborhood_value    

    def train(self, data, verbose=False):
        """

        Trains SOM with training data.
        - data: NumPy array containing input vectors.
        - verbose: Optional parameter to print training progress.
        """

        if self.weights is None:
            raise ValueError("Weights have not been initialized. Call 'random_weights_init' before training.")
        
        if data.shape[1] != self.input_dim:
            raise ValueError(f"Input data dimension {data.shape[1]} does not match the SOM input dimension {self.input_dim}.")
        
        if len(data) == 0:
            raise ValueError("Training data cannot be empty.")
        
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError("Grid size must be positive.")
        
        epochs = self.max_iterations

        for epoch in range(epochs):

            current_learning_rate = self.learning_rate * (1 - epoch / epochs)
            current_learning_rate = max(current_learning_rate, self.min_learning_rate)  
            current_neighborhood_radius = self.initial_radius * (1 - epoch / epochs)
            current_neighborhood_radius = max(current_neighborhood_radius, self.min_radius)

            if verbose and epoch % 10 == 0:
                 print(f"Epoch {epoch}/{epochs} - Learning rate: {current_learning_rate}, Radius: {current_neighborhood_radius}")

            for input_vector in data:
                
                winner_i, winner_j = self.winner(input_vector)

                self.update_weights(input_vector, winner_i, winner_j, current_learning_rate, current_neighborhood_radius, epoch)
            

    def update_weights(self, input_vector, winner_i, winner_j, learning_rate, neighborhood_radius, epoch):
        """
        Updates the weights of neurons based on the input vector, winner, and neighborhood.
        - input_vector: The input data to update the weights with.
        - winner_i, winner_j: The coordinates of the winning neuron (BMU).
        - learning_rate: The learning rate for the current epoch.
        - neighborhood_radius: The neighborhood radius for the current epoch.
        - epoch: The current epoch number.
        """
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                 neighborhood_value = self.neighborhood_function(winner_i, winner_j, i, j, epoch)

                 self.weights[i, j] += neighborhood_value * learning_rate * (input_vector - self.weights[i, j])
        

    def distance_map(self):
     """
         Computes the distance map for the SOM grid.
         This represents the average distance between each neuron and its neighbors.
         - return: A 2D NumPy array of the same shape as the SOM grid, containing the average distances.
     """

     distance_map = np.zeros((self.grid_size[0], self.grid_size[1]))

     for i in range(self.grid_size[0]):
         for j in range(self.grid_size[1]):
             neighbors = []

             if i > 0:  
                 neighbors.append(self.weights[i - 1, j])
             if i < self.grid_size[0] - 1:  
                 neighbors.append(self.weights[i + 1, j])
             if j > 0:  
                 neighbors.append(self.weights[i, j - 1])
             if j < self.grid_size[1] - 1:  
                 neighbors.append(self.weights[i, j + 1])

             if neighbors:
                 current_weight = self.weights[i, j]
                 distances = [np.linalg.norm(current_weight - neighbor) for neighbor in neighbors]
                 distance_map[i, j] = np.mean(distances)

     return distance_map

    def win_map(self, data):
     """
      Maps each input vector in the data to its winning neuron (BMU).
      - data: NumPy array containing input vectors.
      - return: A dictionary where keys are neuron coordinates (e.g., (5, 3))
              and values are lists of input vectors assigned to those neurons.
     """

     if self.weights is None:
            raise ValueError("Weights are not initialized. Call 'random_weights_init' first.")
     if not isinstance(data, (list, np.ndarray)):
            raise ValueError("Data should be a list or numpy array of input vectors.")
     
     winner_map = {} 

     for input_vector in data:
            winner_i, winner_j = self.winner(input_vector)

            if (winner_i, winner_j) not in winner_map:
                winner_map[(winner_i, winner_j)] = []

            winner_map[(winner_i, winner_j)].append(input_vector)

     return winner_map


    def visualize(self, data, labels=None, markers=["o", "s"], colors=["r", "g"]):
        """
    Visualizes the SOM grid with the distance map (MID) and data points.
    - data: Input data used for training the SOM.
    - labels: Optional labels for the data points (for coloring/marking).
    - markers: Marker styles for different classes (default: 'o' and 's').
    - colors: Marker edge colors for different classes (default: 'r' and 'g').
     """
        if self.weights is None:
            raise ValueError("Weights are not initialized. Train the SOM first.")

        if labels is not None and len(data) != len(labels):
            raise ValueError("Length of data and labels must be the same.")

        bone()
        pcolor(self.distance_map().T)
        colorbar()

        for i, x in enumerate(data):
            winner = self.winner(x)
            marker = markers[labels[i]] if labels is not None else "o"
            color = colors[labels[i]] if labels is not None else "b"
            plot(
                winner[0] + 0.5,
                winner[1] + 0.5,
                marker,
                markeredgecolor=color,
                markerfacecolor="None",
                markersize=10,
                markeredgewidth=2,
            )

        show()

class TestSOM:
    def __init__(self):
        self.som = SOM(grid_size=(10, 10), input_dim=3, max_iterations=100)
        self.sample_data = np.random.rand(100, 3)  

    @staticmethod
    def test_constructor():
        """
        Tests the constructor of the SOM class to ensure correct initialization 
        and proper error handling for invalid inputs.
        """
        # 1. Test valid constructor
        try: 
            som = SOM(grid_size=(10, 10), input_dim=3, max_iterations=100)
            assert som.grid_size == (10,10)
            assert som.input_dim == 3
            assert som.max_iterations == 100
            assert som.learning_rate == 0.1
            assert som.distance_metric == "euclidean"
            print("Test valid constructor: PASSED")
        except Exception as e:
            print("Test valid constructor: FAILED", str(e))
        
        # 2. Test invalid grid size (negative or zero)

        try:
            som = SOM(grid_size=(0, 10),input_dim=3)
            print("Test invalid grid size (zero): FAILED")

        except ValueError as e:
            assert "Grid size dimensions must be positive integers" in str(e)
            print("Test invalid grid size (zero): PASSED")

        try:
            som = SOM(grid_size=(-10, 10),input_dim=3)
            print("Test invalid grid size (negative): FAILED")

        except ValueError as e:
            assert "Grid size dimensions must be positive integers" in str(e)
            print("Test invalid grid size (negative): PASSED")

        # 3. Test invalid input dimension (negative or zero)

        try:
            som = SOM(grid_size=(10, 10),input_dim=0)
            print("Test invalid input size (zero): FAILED")

        except ValueError as e:
            print("Test invalid input size (zero): PASSED")

        try:
            som = SOM(grid_size=(10, 10),input_dim=-2)
            print("Test invalid input size (negative): FAILED")

        except ValueError as e:
            print("Test invalid input size (negative): PASSED")

        # 4. Test invalid max_iterations (negative or zero)

        try:
            som = SOM(grid_size=(10, 10), input_dim=3, max_iterations=100)
            print("Test invalid max_iterations (zero): FAILED")
        except:
            print("Test invalid max_iterations (zero): PASSED")

        try:
            som = SOM(grid_size=(10, 10), input_dim=3, max_iterations=-100)
            print("Test invalid max_iterations (negative): FAILED")
        except:
            print("Test invalid max_iterations (negative): PASSED")   

        # 5. Test invalid distance metric

        try: 
            som = SOM(grid_size=(10, 10), input_dim=3, distance_metric="invalid_metric")
            print("Test invalid distance metric: FAILED")
        except:
            print("Test invalid distance metric: PASSED")



    @staticmethod
    def test_random_weights_init():
        """
        Tests the random_weights_init method of the SOM class to ensure correct initialization of weights.
        """
        # 1. Test if weights are initialized correctly
        try:
            som = SOM(grid_size=(10, 10), input_dim=3)
            som.random_weights_init()
            assert som.weights is not None, "Weights should not be None after initialization."
            assert som.weights.shape == (10, 10, 3), "Weights shape should match the grid size and input dimension."
            print("Test weights initialization: PASSED")
        except Exception as e:
            print("Test weights initialization: FAILED", str(e))

        # 2. Test deterministic initialization with seed

        try:
            som1 = SOM(grid_size=(10, 10), input_dim=3)
            som2 = SOM(grid_size=(10, 10), input_dim=3)
            som1.random_weights_init(seed=35)
            som2.random_weights_init(seed=35)

            assert np.array_equal(som1.weights,som2.weights), "Weights should be identical when initialized with the same seed."
            print("Test deterministic initialization with seed: PASSED")
        except Exception as e:
            print("Test deterministic initialization with seed: FAILED", str(e))

         # 3. Test random initialization without seed (non-deterministic)

        try:
            som1 = SOM(grid_size=(10,10,3),input_dim=3)
            som2 = SOM(grid_size=(10,10,3),input_dim=3)
            som1.random_weights_init()
            som2.random_weights_init()

            assert not np.array_equal(som1.weights,som2.weights), "Weights should differ when no seed is provided."
            print("Test non-deterministic initialization: PASSED")

        except Exception as e:
            print("Test non-deterministic initialization: FAILED", str(e))

        # 4. Test weights initialization with invalid seed type
        
        try:
            som = SOM(grid_size=(10,10,3),input_dim=3)
            som.random_weights_init(seed="invalid_seed")
            print("Test invalid seed type: FAILED")
        except TypeError as e:
            print("Test invalid seed type: PASSED")
        except Exception as e:
            print("Test invalid seed type: FAILED", str(e))


    @staticmethod
    def test_activation_distance():
        """
        Tests the activation_distance method of the SOM class to ensure correct distance calculations.
        """

        # 1. Test Euclidean distance
        try:
            som = SOM(grid_size=(5, 5), input_dim=3, distance_metric="euclidean")
            som.random_weights_init(seed=42)
            input_vector = np.array([0.5, 0.5, 0.5])
            distances = som.activation_distance(input_vector)

            assert distances.shape == (5,5), "Distance matrix shape should match grid size."
            expected_distances = np.sqrt(np.sum((som.weights - input_vector.reshape(1, 1, 3)) ** 2, axis=2))
            assert np.allclose(distances, expected_distances), "Euclidean distances are not calculated correctly."
            print("Test Euclidean distance: PASSED")
        except Exception as e:
             print("Test Euclidean distance: FAILED", str(e))

        # 2. Test Manhattan distance
        try:
            som = SOM(grid_size=(5, 5), input_dim=3, distance_metric="manhattan")
            som.random_weights_init(seed=42)
            input_vector = np.array([0.5, 0.5, 0.5])
            distances = som.activation_distance(input_vector)

            expected_distances = np.sum(np.abs(som.weights - input_vector.reshape(1, 1, 3)), axis=2)
            assert np.allclose(distances, expected_distances), "Manhattan distances are not calculated correctly."
            print("Test Manhattan distance: PASSED")
        except Exception as e:
             print("Test Manhattan distance: FAILED", str(e))

        # 3. Test Cosine distance
        try:
            som = SOM(grid_size=(5, 5), input_dim=3, distance_metric="cosine")
            som.random_weights_init(seed=42)
            input_vector = np.array([0.5, 0.5, 0.5])
            distances = som.activation_distance(input_vector)

            dot_product = np.sum(input_vector * som.weights, axis=2)
            norm_input = np.linalg.norm(input_vector)
            norm_weights = np.linalg.norm(som.weights, axis=2)
            expected_distances = 1 - (dot_product / (norm_input * norm_weights + 1e-10))
            assert np.allclose(distances, expected_distances), "Cosine distances are not calculated correctly."
            print("Test Cosine distance: PASSED")
        except Exception as e:
             print("Test Cosine distance: FAILED", str(e))


        # 4. Test Invalid Distance Metric

        try:
            som = SOM(grid_size=(5, 5), input_dim=3, distance_metric="invalid_metric")
            print("Test invalid distance metric: FAILED")
        except ValueError as e:
            print("Test invalid distance metric: PASSED")
        except Exception as e:
            print("Test invalid distance metric: FAILED", str(e))

    def test_winner(self):
        """
        Tests the winner method of the SOM class to ensure the correct BMU is identified.
        """

        # 1. Test with Euclidean distance
        try:
            som = SOM(grid_size=(5, 5), input_dim=3, distance_metric="euclidean")
            som.random_weights_init(seed=42)
            input_vector = np.array([0.5, 0.5, 0.5])
            winner = som.winner(input_vector)

            assert isinstance(winner, tuple), "Winner should be a tuple of coordinates."
            assert len(winner) == 2, "Winner tuple should contain two elements (i, j)."
            distances = np.sqrt(np.sum((som.weights - input_vector.reshape(1, 1, 3)) ** 2, axis=2))
            expected_winner = np.unravel_index(np.argmin(distances), som.grid_size)
            assert winner == expected_winner, "Winner neuron coordinates are incorrect."
            print("Test Euclidean winner: PASSED")
        except Exception as e:
            print("Test Euclidean winner: FAILED", str(e))
        
        # 2. Test with Manhattan distance

        try:
            som = SOM(grid_size=(5, 5), input_dim=3, distance_metric="manhattan")
            som.random_weights_init(seed=42)
            input_vector = np.array([0.5, 0.5, 0.5])
            winner = som.winner(input_vector)

            distances = np.sum(np.abs(som.weights - input_vector.reshape(1, 1, 3)), axis=2)
            expected_winner = np.unravel_index(np.argmin(distances), som.grid_size)
            assert winner == expected_winner, "Winner neuron coordinates for Manhattan distance are incorrect."
            print("Test Manhattan winner: PASSED")
        except Exception as e:
            print("Test Manhattan winner: FAILED", str(e))

        # 3. Test with empty weights

        try:
            som = SOM(grid_size=(5, 5), input_dim=3)
            input_vector = np.array([0.5, 0.5, 0.5])
            som.weights = None
            som.winner(input_vector)
            print("Test empty weights: FAILED")
        except ValueError as e:
            print("Test empty weights: PASSED")
        except Exception as e:
            print("Test empty weights: FAILED", str(e))
        
        # 4. Test with invalid grid size

        try:
            som = SOM(grid_size=(0, 0), input_dim=3)
            som.random_weights_init(seed=42)
            input_vector = np.array([0.5, 0.5, 0.5])
            som.winner(input_vector)
            print("Test invalid grid size: FAILED")
        except ValueError as e:
            print("Test invalid grid size: PASSED")
        except Exception as e:
            print("Test invalid grid size: FAILED", str(e))

    @staticmethod
    def test_win_map():
        try:
            som = SOM(grid_size=(5, 5), input_dim=3)
            som.random_weights_init(seed=42)
            data = np.random.rand(10, 3)
            win_map_result = som.win_map(data)

            assert isinstance(win_map_result, dict), "Win map result should be a dictionary."
            for key, value in win_map_result.items():
                assert isinstance(key, tuple), "Neuron coordinates should be a tuple."
                for vec in value:
                    assert vec.shape == (3,), "Each input vector should have the correct shape."
            print("Test win_map function: PASSED")
        except Exception as e:
            print("Test win_map function: FAILED", str(e))

    @staticmethod
    def test_train():
        try:
            som = SOM(grid_size=(5, 5), input_dim=3, max_iterations=10)
            som.random_weights_init(seed=42)
            data = np.random.rand(20, 3)
            initial_weights = np.copy(som.weights)
            som.train(data)

            assert not np.array_equal(som.weights, initial_weights), "Weights should change after training."
            print("Test train function: PASSED")
        except Exception as e:
            print("Test train function: FAILED", str(e))

    @staticmethod
    def test_update_weights():
        try:
            som = SOM(grid_size=(3, 3), input_dim=2)
            som.random_weights_init(seed=42)
            input_vector = np.array([0.5, 0.5])
            winner = (1, 1)
            som.update_weights(input_vector, *winner, learning_rate=0.5, neighborhood_radius=1.0, epoch=0)

            assert not np.allclose(som.weights, np.random.rand(3, 3, 2)), "Weights should update based on the input vector."
            print("Test update_weights function: PASSED")
        except Exception as e:
            print("Test update_weights function: FAILED", str(e))


    @staticmethod
    def test_distance_map():
        try:
            som = SOM(grid_size=(5, 5), input_dim=3)
            som.random_weights_init(seed=42)
            distance_map = som.distance_map()

            assert distance_map.shape == (5, 5), "Distance map shape should match grid size."
            assert np.all(distance_map >= 0), "Distance map values should be non-negative."
            print("Test distance_map function: PASSED")
        except Exception as e:
            print("Test distance_map function: FAILED", str(e))

